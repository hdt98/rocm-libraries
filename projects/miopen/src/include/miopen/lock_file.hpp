/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_LOCK_FILE_HPP_
#define GUARD_MIOPEN_LOCK_FILE_HPP_

#include <miopen/errors.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/file_lock.hpp>
#include <miopen/logger.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

namespace miopen {

class FSLockFile
{
public:
    enum class LockType { None, Exclusive, Shared };

    FSLockFile() : lock_held(false), lock_type(LockType::None) {}
    FSLockFile(const fs::path& path_) : lock_held(false), lock_type(LockType::None)
    {
        lockfile_path = path_.string() + ".fslock";

        auto process_id = generate_process_identifier();
        unique_handle = lockfile_path.string() + "." + process_id;

        // Make reader_file unique per object instance using object address
        std::ostringstream oss;
        oss << lockfile_path.string() << "/" << process_id << "."
            << std::this_thread::get_id() << "."
            << static_cast<const void*>(this) << ".reader";
        reader_file = oss.str();
    }

    // Prevent copying and moving (atomic<bool> is not movable)
    FSLockFile(const FSLockFile&) = delete;
    FSLockFile& operator=(const FSLockFile&) = delete;
    FSLockFile(FSLockFile&&) = delete;
    FSLockFile& operator=(FSLockFile&&) = delete;

    bool timed_lock(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        return timed_lock_impl([this]() { return try_lock_hardlink(); }, abs_time, "Lock");
    }

    // Duration-based exclusive lock (std::shared_timed_mutex compatibility)
    template <class TDuration>
    bool try_lock_for(TDuration duration)
    {
        auto abs_time = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        return timed_lock(abs_time);
    }

    // Time-point exclusive lock (std::shared_timed_mutex compatibility)
    template <class TPoint>
    bool try_lock_until(TPoint point)
    {
        return timed_lock(point);
    }

    void lock()
    {
        blocking_lock_impl([this]() { return try_lock_hardlink(); }, "Lock");
    }

    bool try_lock()
    {
        bool acquired = false;
        acquired      = try_lock_hardlink();
        if(acquired)
            MIOPEN_LOG_I2("Lock < " << lockfile_path.string());
        return acquired;
    }

    bool clear_stale_lock()
    {
        // Check if lockfile exists before accessing it
        std::error_code ec;
        if(!fs::exists(lockfile_path, ec) || ec)
            return false;

        auto last_write_time = fs::last_write_time(lockfile_path, ec);
        if(ec)
            return false; // File was deleted between exists check and last_write_time

        auto now = fs::file_time_type::clock::now();
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_write_time);

        if(now < last_write_time)
        {
            MIOPEN_LOG_I2("Clocks desyncronized, Lock write time is later than present < " << lockfile_path.string()
                                                   << ", Age(ms): " << age.count());
            if(last_write_seen != last_write_time)
            {
                last_write_seen = last_write_time;
                now_at_write    = now;
            }
            age = std::chrono::duration_cast<std::chrono::milliseconds>(now - now_at_write);
        }

        if(age > STALE_TIMEOUT)
        {
            MIOPEN_LOG_I2("Removing Stale Lock < " << lockfile_path.string()
                                                   << ", Age(ms): " << age.count());
            std::error_code remove_ec;
            fs::remove(lockfile_path, remove_ec);
            // Return true even if remove failed - another process may have removed it
            return true;
        }
        return false;
    }

    bool try_lock_hardlink()
    {
        std::error_code ec;

        // Check if lockfile_path exists as a directory (shared locks active)
        if(fs::is_directory(lockfile_path, ec))
        {
            // Shared locks are active - check if any are fresh
            clean_stale_readers();
            if(has_active_readers())
            {
                return false; // Cannot acquire exclusive lock while readers exist
            }
            // Directory exists but empty - try to remove it
            fs::remove(lockfile_path, ec);
        }
        else
        {
            clear_stale_lock();
        }

        if(!fs::exists(unique_handle))
        {
            if(!std::ofstream{unique_handle})
                MIOPEN_THROW("Error creating file <" + unique_handle + "> for locking.");
        }

        std::error_code perm_ec;
        fs::permissions(unique_handle, fs::perms::all, perm_ec);
        // Ignore permission errors - file might have been removed by another process

        create_hard_link(unique_handle, lockfile_path, ec);
        if(ec.value() == 0)
        {
            if(fs::hard_link_count(unique_handle) == 2)
            {
                // Double-check no readers appeared while we were acquiring lock
                if(fs::is_directory(lockfile_path, ec) || has_active_readers())
                {
                    // Clean up our lock
                    fs::remove(lockfile_path, ec);
                    return false;
                }

                lock_type = LockType::Exclusive;
                lock_held.store(true, std::memory_order_release);
                refresh_thread = std::thread([this]() { this->refresh_lock(); });
                return true;
            }
        }
        return false;
    }

    void unlock()
    {
        MIOPEN_LOG_I2("Unlock < " << lockfile_path.string());

        // Remove lock files immediately to allow other threads to acquire the lock
        remove_file_with_log(lockfile_path, "lockfile");
        remove_file_with_log(unique_handle, "unique handle");

        stop_refresh_thread();
        lock_type = LockType::None;
    }

    fs::path get_unique_handle() { return unique_handle; }

    // Shared lock methods
    bool try_lock_shared()
    {
        // Key insight: lockfile_path is used for BOTH exclusive and shared locks
        // - Exclusive lock creates lockfile_path as a regular file (hardlink)
        // - Shared lock creates lockfile_path as a directory with reader files inside
        // The filesystem enforces mutual exclusion - can't be both file and directory!

        std::error_code ec;

        // Atomically create readers directory - fails if lockfile_path is a file (exclusive lock)
        fs::create_directories(lockfile_path, ec);
        if(ec)
        {
            // Failed to create directory - might be a stale exclusive lock (regular file)
            // Try to clean up stale exclusive lock
            if(fs::exists(lockfile_path, ec) && !ec && !fs::is_directory(lockfile_path, ec))
            {
                // It's a regular file - check if stale
                if(clear_stale_lock())
                {
                    // Stale lock removed, try again
                    fs::create_directories(lockfile_path, ec);
                    if(ec)
                        return false;
                }
                else
                {
                    // Active exclusive lock
                    return false;
                }
            }
            else
            {
                // Other error
                return false;
            }
        }

        // Verify it's actually a directory (paranoid check)
        if(!fs::is_directory(lockfile_path, ec) || ec)
        {
            return false;
        }

        // Create our reader file inside the directory
        if(!std::ofstream{reader_file})
        {
            MIOPEN_LOG_W("Error creating reader file: " << reader_file);
            return false;
        }
        fs::permissions(reader_file, fs::perms::all, ec);

        // No race condition! The directory can't become a file while we have a file inside it.
        // The filesystem prevents removing a non-empty directory.

        // Success - start refresh thread
        lock_type = LockType::Shared;
        lock_held.store(true, std::memory_order_release);
        refresh_thread = std::thread([this]() { this->refresh_lock_shared(); });

        MIOPEN_LOG_I2("Shared Lock < " << reader_file);
        return true;
    }

    void lock_shared()
    {
        blocking_lock_impl([this]() { return try_lock_shared(); }, "Shared Lock");
    }

    bool timed_lock_shared(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        return timed_lock_impl([this]() { return try_lock_shared(); }, abs_time, "Shared Lock");
    }

    // Duration-based shared lock (std::shared_timed_mutex compatibility)
    template <class TDuration>
    bool try_lock_shared_for(TDuration duration)
    {
        auto abs_time = std::chrono::steady_clock::now() +
                       std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        return timed_lock_shared(abs_time);
    }

    // Time-point shared lock (std::shared_timed_mutex compatibility)
    template <class TPoint>
    bool try_lock_shared_until(TPoint point)
    {
        return timed_lock_shared(point);
    }

    void unlock_shared()
    {
        MIOPEN_LOG_I2("Unlock Shared < " << reader_file);

        // Remove reader file immediately
        remove_file_with_log(reader_file, "reader file");

        stop_refresh_thread();

        // Clean up the readers directory if it's empty
        // This allows exclusive locks to be acquired after all shared locks are released
        std::error_code ec;
        if(fs::is_empty(lockfile_path, ec) && !ec)
        {
            remove_file_with_log(lockfile_path, "empty readers directory");
            MIOPEN_LOG_I2("Removed empty readers directory < " << lockfile_path.string());
        }

        lock_type = LockType::None;
    }

private:
    // Configuration constants
    static constexpr auto STALE_TIMEOUT = std::chrono::milliseconds(20);
    static constexpr auto REFRESH_FREQUENCY = std::chrono::milliseconds(4);
    static constexpr auto POLL_INTERVAL = std::chrono::microseconds(100);

    // Helper: Generate unique process identifier
    std::string generate_process_identifier() const
    {
        return sysinfo::GetSystemHostname() + "." + std::to_string(getpid());
    }

    // Helper: Unified blocking lock implementation
    template<typename TryLockFunc>
    void blocking_lock_impl(TryLockFunc try_lock_fn, const std::string& lock_name)
    {
        bool acquired = false;
        MIOPEN_LOG_I2("Attempting " << lock_name << " < " << lockfile_path.string());
        while(!acquired)
        {
            acquired = try_lock_fn();
            if(!acquired)
                std::this_thread::sleep_for(POLL_INTERVAL);
            else
                MIOPEN_LOG_I2(lock_name << " < " << lockfile_path.string());
        }
    }

    // Helper: Unified timed lock implementation
    template<typename TryLockFunc>
    bool timed_lock_impl(TryLockFunc try_lock_fn,
                         const std::chrono::time_point<std::chrono::steady_clock>& abs_time,
                         const std::string& lock_name)
    {
        auto now = std::chrono::steady_clock::now();
        bool acquired = false;
        MIOPEN_LOG_I2("Attempting " << lock_name << " < " << lockfile_path.string());

        while(!acquired && now < abs_time)
        {
            now = std::chrono::steady_clock::now();
            acquired = try_lock_fn();

            if(!acquired && now < abs_time)
                std::this_thread::sleep_for(POLL_INTERVAL);
            else if(acquired)
                MIOPEN_LOG_I2(lock_name << " < " << lockfile_path.string());
        }
        return acquired;
    }

    // Helper: Unified refresh thread logic
    void refresh_file(const fs::path& file_to_refresh, const std::string& log_name)
    {
        MIOPEN_LOG_I2(log_name << " Refresh Active < " << file_to_refresh.string());
        auto last_refresh = fs::file_time_type::clock::now();
        auto age = REFRESH_FREQUENCY;

        std::unique_lock<std::mutex> lock(refresh_mutex);
        while(lock_held.load(std::memory_order_acquire))
        {
            if(age >= REFRESH_FREQUENCY)
            {
                std::error_code ec;
                fs::last_write_time(file_to_refresh, fs::file_time_type::clock::now(), ec);
                if(ec.value() != 0)
                {
                    MIOPEN_LOG_I2("File <" << file_to_refresh << "> "
                                  << " time update failed. Terminating refresh. "
                                  << "Error code: " << ec.value() << ". "
                                  << "Description: '" << ec.message() << "'");
                    lock_held.store(false, std::memory_order_release);
                    return;
                }
                last_refresh = fs::file_time_type::clock::now();
            }

            refresh_cv.wait_for(lock, REFRESH_FREQUENCY,
                [this]() { return !lock_held.load(std::memory_order_acquire); });

            auto now = fs::file_time_type::clock::now();
            age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refresh);
        }
        MIOPEN_LOG_I2(log_name << " Refresh Exit < " << file_to_refresh.string());
    }

    // Helper: Stop refresh thread
    void stop_refresh_thread()
    {
        lock_held.store(false, std::memory_order_release);
        refresh_cv.notify_one();

        if(refresh_thread.joinable())
            refresh_thread.join();
    }

    // Helper: Remove file with error logging
    void remove_file_with_log(const fs::path& file, const std::string& description)
    {
        std::error_code ec;
        fs::remove(file, ec);
        if(ec)
            MIOPEN_LOG_W("Failed to remove " << description << ": " << ec.message());
    }

    // Helper: Iterate over reader files
    template<typename ReaderAction>
    void iterate_readers(ReaderAction action)
    {
        std::error_code ec;

        if(!fs::is_directory(lockfile_path, ec) || ec)
            return;

        for(const auto& entry : fs::directory_iterator(lockfile_path, ec))
        {
            if(ec)
                continue;

            auto last_write = fs::last_write_time(entry.path(), ec);
            if(ec)
                continue;

            auto now = fs::file_time_type::clock::now();
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_write);

            action(entry.path(), age);
        }
    }

    void refresh_lock()
    {
        refresh_file(unique_handle, "Lock");
    }

    void refresh_lock_shared()
    {
        refresh_file(reader_file, "Shared Lock");
    }

    bool has_active_readers()
    {
        bool found_active = false;

        iterate_readers([&](const fs::path&, std::chrono::milliseconds age) {
            if(age <= STALE_TIMEOUT)
                found_active = true;
        });

        return found_active;
    }

    void clean_stale_readers()
    {
        iterate_readers([&](const fs::path& reader_path, std::chrono::milliseconds age) {
            if(age > STALE_TIMEOUT)
            {
                MIOPEN_LOG_I2("Removing stale reader < " << reader_path.string()
                              << ", Age(ms): " << age.count());
                std::error_code ec;
                fs::remove(reader_path, ec);
            }
        });
    }

    std::atomic<bool> lock_held;
    LockType lock_type;
    fs::path lockfile_path;
    fs::path unique_handle;
    fs::path reader_file;
    std::thread refresh_thread;
    // Instance-specific state for clock desync handling
    fs::file_time_type last_write_seen{};
    fs::file_time_type now_at_write{};
    // Synchronization for graceful refresh thread termination
    std::mutex refresh_mutex;
    std::condition_variable refresh_cv;
};

MIOPEN_INTERNALS_EXPORT fs::path LockFilePath(const fs::path& filename_);

/// @deprecated Use FSLockFile instead for new code.
/// LockFile is kept for backward compatibility but is not actively used in MIOpen.
/// All database operations now use FSLockFile for filesystem-based locking.
///
/// LockFile class is a wrapper around miopen::file_lock providing MT-safety.
/// One process should never have more than one instance of this class with same path at the same
/// time. It may lead to undefined behaviour on Windows.
/// Also on windows mutex can be removed because file locks are MT-safe there.
class MIOPEN_INTERNALS_EXPORT LockFile
{
private:
    class PassKey
    {
    };

public:
    LockFile(const fs::path&, PassKey);
    LockFile(const LockFile&)           = delete;
    LockFile operator=(const LockFile&) = delete;

    bool timed_lock(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        access_mutex.lock();
        return flock.timed_lock(abs_time);
    }

    bool timed_lock_shared(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        access_mutex.lock_shared();
        return flock.timed_lock_sharable(abs_time);
    }
    void lock()
    {
        LockOperation("lock", MIOPEN_GET_FN_NAME, [&]() { std::lock(access_mutex, flock); });
    }

    void lock_shared()
    {
        access_mutex.lock_shared();
        try
        {
            LockOperation("shared lock", MIOPEN_GET_FN_NAME, [&]() { flock.lock_sharable(); });
        }
        catch(...)
        {
            access_mutex.unlock();
        }
    }

    bool try_lock()
    {
        return TryLockOperation(
            "lock", MIOPEN_GET_FN_NAME, [&]() { return std::try_lock(access_mutex, flock) != 0; });
    }

    bool try_lock_shared()
    {
        if(!access_mutex.try_lock_shared())
            return false;

        if(TryLockOperation(
               "shared lock", MIOPEN_GET_FN_NAME, [&]() { return flock.try_lock_sharable(); }))
            return true;
        access_mutex.unlock();
        return false;
    }

    void unlock()
    {
        LockOperation("unlock", MIOPEN_GET_FN_NAME, [&]() { flock.unlock(); });
        access_mutex.unlock();
    }

    void unlock_shared()
    {
        LockOperation("unlock shared", MIOPEN_GET_FN_NAME, [&]() { flock.unlock_sharable(); });
        access_mutex.unlock_shared();
    }

    static LockFile& Get(const fs::path& file);

    template <class TDuration>
    bool try_lock_for(TDuration duration)
    {
        if(!access_mutex.try_lock_for(duration))
            return false;

        if(TryLockOperation("timed lock", MIOPEN_GET_FN_NAME, [&]() {
               return flock.timed_lock(ToPTime(duration));
           }))
            return true;
        access_mutex.unlock();
        return false;
    }

    template <class TDuration>
    bool try_lock_shared_for(TDuration duration)
    {
        if(!access_mutex.try_lock_shared_for(duration))
            return false;

        if(TryLockOperation("shared timed lock", MIOPEN_GET_FN_NAME, [&]() {
               return flock.timed_lock_sharable(ToPTime(duration));
           }))
            return true;
        access_mutex.unlock();
        return false;
    }

    template <class TPoint>
    bool try_lock_until(TPoint point)
    {
        return try_lock_for(point - std::chrono::system_clock::now());
    }

    template <class TPoint>
    bool try_lock_shared_until(TPoint point)
    {
        return try_lock_shared_for(point - std::chrono::system_clock::now());
    }

private:
    fs::path path; // For logging purposes
    std::shared_timed_mutex access_mutex;
    miopen::file_lock flock;

    static std::map<fs::path, LockFile>& LockFiles()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::map<fs::path, LockFile> lock_files;
        return lock_files;
    }

    template <class TDuration>
    static std::chrono::time_point<std::chrono::steady_clock> ToPTime(TDuration duration)
    {
        return std::chrono::steady_clock::now() +
               std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    }

    void LogFlockError(const std::exception& ex,
                       const std::string& operation,
                       const std::string_view from) const
    {
        // clang-format off
        MIOPEN_LOG_E_FROM(from, "File <" << path << "> " << operation << " failed. "
                                "Description: '" << ex.what() << "'");
        // clang-format on
    }

    void LockOperation(const std::string& op_name,
                       const std::string_view from,
                       std::function<void()>&& op)
    {
        try
        {
            op();
        }
        catch(const std::exception& ex)
        {
            LogFlockError(ex, op_name, from);
            throw;
        }
    }

    bool TryLockOperation(const std::string& op_name,
                          const std::string_view from,
                          std::function<bool()>&& op)
    {
        try
        {
            if(op())
                return true;
            MIOPEN_LOG_W("File <" << path << "> " << op_name << " timed out.");
            return false;
        }
        catch(const std::exception& ex)
        {
            LogFlockError(ex, op_name, from);
            return false;
        }
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_LOCK_FILE_HPP_
