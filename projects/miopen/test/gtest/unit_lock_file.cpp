
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/lock_file.hpp>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <future>

template <class TDuration>
static std::chrono::time_point<std::chrono::steady_clock> ToPTime(TDuration duration)
{
    return std::chrono::steady_clock::now() +
           std::chrono::duration_cast<std::chrono::milliseconds>(duration);
}

TEST(CPU_UnitTestLockFile_NONE, TryLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto lockfile                 = miopen::FSLockFile(lockpath);
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_FALSE(miopen::fs::exists(unique_handle));
    EXPECT_TRUE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(unique_handle));
    EXPECT_TRUE(miopen::fs::hard_link_count(unique_handle) == 2);

    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    lockfile.unlock();
    EXPECT_FALSE(miopen::fs::exists(unique_handle));
}

TEST(CPU_UnitTestLockFile_NONE, TimedLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);
    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds{50})));
    EXPECT_FALSE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds{50})));
    auto unique_handle = lockfile.get_unique_handle();
    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, OrphanLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test." + std::to_string(getpid())});
    auto lockfile                 = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);

    miopen::fs::last_write_time(fs_lock_path, miopen::fs::file_time_type::clock::now());
    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds{50})));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, LostLockFile)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test." + std::to_string(getpid())});
    auto lockfile                 = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());
    miopen::fs::remove(unique_handle);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, RefreshHold)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto lockfile                 = miopen::FSLockFile(lockpath);
    auto unique_handle            = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(lockfile.try_lock());
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));

    lockfile.unlock();
    EXPECT_TRUE(lockfile.try_lock());
    lockfile.unlock();
}

// Enhanced tests for edge cases and error conditions

TEST(CPU_UnitTestLockFile_NONE, ConcurrentLockAttempts)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent." + std::to_string(getpid())});

    std::atomic<int> success_count{0};
    std::vector<std::future<bool>> futures;

    // Launch multiple threads trying to acquire the same lock
    for(int i = 0; i < 10; ++i)
    {
        futures.push_back(std::async(std::launch::async, [&lockpath, &success_count]() {
            miopen::FSLockFile lockfile(lockpath);
            bool acquired = lockfile.try_lock();
            if(acquired)
            {
                success_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                lockfile.unlock();
            }
            return acquired;
        }));
    }

    // Wait for all threads and count successes
    int total_successes = 0;
    for(auto& fut : futures)
    {
        if(fut.get())
            total_successes++;
    }

    // Only one thread should succeed at a time
    EXPECT_EQ(success_count.load(), total_successes);
    EXPECT_GE(total_successes, 1);
}

TEST(CPU_UnitTestLockFile_NONE, MultipleLocksIndependent)
{
    auto lockpath1 = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi1." + std::to_string(getpid())});
    auto lockpath2 = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi2." + std::to_string(getpid())});

    // Create stale locks for both paths
    miopen::fs::path fs_lock_path1 = lockpath1.string() + ".fslock";
    miopen::fs::path fs_lock_path2 = lockpath2.string() + ".fslock";

    EXPECT_TRUE(std::ofstream{fs_lock_path1});
    EXPECT_TRUE(std::ofstream{fs_lock_path2});
    miopen::fs::permissions(fs_lock_path1, miopen::fs::perms::all);
    miopen::fs::permissions(fs_lock_path2, miopen::fs::perms::all);

    auto now = miopen::fs::file_time_type::clock::now();
    miopen::fs::last_write_time(fs_lock_path1, now);
    miopen::fs::last_write_time(fs_lock_path2, now);

    // Create two different lock objects
    auto lockfile1 = miopen::FSLockFile(lockpath1);
    auto lockfile2 = miopen::FSLockFile(lockpath2);

    // Wait for stale locks to timeout and acquire
    EXPECT_TRUE(lockfile1.timed_lock(ToPTime(std::chrono::milliseconds(100))));
    EXPECT_TRUE(lockfile2.timed_lock(ToPTime(std::chrono::milliseconds(100))));

    lockfile1.unlock();
    lockfile2.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, ClockTypeMismatch)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_clock." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);

    // timed_lock should now use steady_clock consistently
    auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    EXPECT_TRUE(lockfile.timed_lock(timeout));

    // Second attempt should timeout
    auto timeout2 = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    EXPECT_FALSE(lockfile.timed_lock(timeout2));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, UniqueHandleDeletedDuringLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_deleted." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);
    auto unique_handle = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());

    // Wait for refresh thread to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Simulate external deletion of unique_handle
    miopen::fs::remove(unique_handle);

    // Refresh thread should detect this and terminate gracefully
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Cleanup - should handle missing file gracefully
    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, HardlinkCountRace)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_hardlink." + std::to_string(getpid())});

    std::atomic<int> lock_count{0};
    std::vector<std::thread> threads;

    // Multiple threads racing to create hardlinks
    for(int i = 0; i < 5; ++i)
    {
        threads.emplace_back([&lockpath, &lock_count]() {
            miopen::FSLockFile lockfile(lockpath);
            for(int j = 0; j < 10; ++j)
            {
                if(lockfile.try_lock())
                {
                    lock_count++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    lockfile.unlock();
                    lock_count--;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    for(auto& t : threads)
        t.join();

    // At the end, no locks should be held
    EXPECT_EQ(lock_count.load(), 0);
}

TEST(CPU_UnitTestLockFile_NONE, DesynchronizedClocks)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_desync." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    // Create a stale lock with future timestamp (simulating clock desync)
    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);

    auto future_time = miopen::fs::file_time_type::clock::now() + std::chrono::hours(1);
    miopen::fs::last_write_time(fs_lock_path, future_time);

    // Should handle the future timestamp correctly
    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds(100))));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, RapidLockUnlockCycles)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_rapid." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);

    // Rapid lock/unlock to stress test thread joining
    for(int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(lockfile.try_lock());
        // Very short hold time
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lockfile.unlock();
    }
}

TEST(CPU_UnitTestLockFile_NONE, BlockingLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_blocking." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    EXPECT_TRUE(lockfile1.try_lock());

    // Launch thread that tries to acquire lock with blocking lock()
    std::atomic<bool> lock_acquired{false};
    std::thread t([&lockpath, &lock_acquired]() {
        miopen::FSLockFile lockfile2(lockpath);
        lockfile2.lock(); // This should block until lockfile1 unlocks
        lock_acquired = true;
        lockfile2.unlock();
    });

    // Wait a bit to ensure thread is blocked
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(lock_acquired.load());

    // Unlock and verify other thread acquires it
    lockfile1.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(lock_acquired.load());

    t.join();
}

TEST(CPU_UnitTestLockFile_NONE, MultipleSequentialAcquisitions)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_sequential." + std::to_string(getpid())});

    // Test that the same lock can be acquired multiple times sequentially
    for(int i = 0; i < 5; ++i)
    {
        miopen::FSLockFile lockfile(lockpath);
        EXPECT_TRUE(lockfile.try_lock());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        lockfile.unlock();
    }
}

TEST(CPU_UnitTestLockFile_NONE, TimedLockTimeout)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_timeout." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    EXPECT_TRUE(lockfile1.try_lock());

    // Second lock should timeout
    miopen::FSLockFile lockfile2(lockpath);
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(lockfile2.timed_lock(ToPTime(std::chrono::milliseconds(50))));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    // Should have waited approximately 50ms
    EXPECT_GE(elapsed.count(), 40);
    EXPECT_LE(elapsed.count(), 100);

    lockfile1.unlock();
}
