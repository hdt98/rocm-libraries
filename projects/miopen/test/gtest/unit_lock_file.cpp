
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

// ========== Basic Exclusive Lock Tests ==========

TEST(CPU_UnitTestLockFile_NONE, BasicLockFileCreationAndCleanup)
{
    // Test filesystem artifact creation and cleanup
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

TEST(CPU_UnitTestLockFile_NONE, StaleExclusiveLockCleanup)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_stale." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    auto lockfile                 = miopen::FSLockFile(lockpath);

    // Create stale lock
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
        miopen::fs::path{"/tmp/config/miopen/test_lost." + std::to_string(getpid())});
    auto lockfile      = miopen::FSLockFile(lockpath);
    auto unique_handle = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());
    miopen::fs::remove(unique_handle);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, RapidLockUnlockCyclesBothTypes)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_rapid." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);

    // Test exclusive locks
    for(int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(lockfile.try_lock());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lockfile.unlock();
    }

    // Test shared locks
    for(int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(lockfile.try_lock_shared());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lockfile.unlock_shared();
    }
}

TEST(CPU_UnitTestLockFile_NONE, BlockingLockBothTypes)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_blocking." + std::to_string(getpid())});

    // Test blocking exclusive lock
    {
        miopen::FSLockFile lockfile1(lockpath);
        EXPECT_TRUE(lockfile1.try_lock());

        std::atomic<bool> lock_acquired{false};
        std::thread t([&lockpath, &lock_acquired]() {
            miopen::FSLockFile lockfile2(lockpath);
            lockfile2.lock();
            lock_acquired = true;
            lockfile2.unlock();
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(lock_acquired.load());

        lockfile1.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_TRUE(lock_acquired.load());

        t.join();
    }

    // Test blocking shared lock
    {
        miopen::FSLockFile exclusive_lock(lockpath);
        EXPECT_TRUE(exclusive_lock.try_lock());

        std::atomic<bool> shared_acquired{false};
        std::thread t([&lockpath, &shared_acquired]() {
            miopen::FSLockFile shared_lock(lockpath);
            shared_lock.lock_shared();
            shared_acquired = true;
            shared_lock.unlock_shared();
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(shared_acquired.load());

        exclusive_lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_TRUE(shared_acquired.load());

        t.join();
    }
}

// ========== Concurrency Tests ==========

TEST(CPU_UnitTestLockFile_NONE, HardlinkCountRace)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_hardlink." + std::to_string(getpid())});

    std::atomic<int> lock_count{0};
    std::vector<std::thread> threads;

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

    EXPECT_EQ(lock_count.load(), 0);
}

TEST(CPU_UnitTestLockFile_NONE, MassiveConcurrentLoad)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_massive_load." + std::to_string(getpid())});

    std::atomic<int> total_acquires{0};
    std::vector<std::thread> threads;

    for(int i = 0; i < 20; ++i)
    {
        threads.emplace_back([&, i]() {
            for(int j = 0; j < 10; ++j)
            {
                miopen::FSLockFile lock(lockpath);
                bool exclusive = (i % 2 == 0);

                if(exclusive)
                {
                    if(lock.try_lock())
                    {
                        total_acquires++;
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                        lock.unlock();
                    }
                }
                else
                {
                    if(lock.try_lock_shared())
                    {
                        total_acquires++;
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                        lock.unlock_shared();
                    }
                }

                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    for(auto& t : threads)
        t.join();

    EXPECT_GT(total_acquires.load(), 50);
}

// ========== Clock and Timeout Tests ==========

TEST(CPU_UnitTestLockFile_NONE, ClockTypeMismatch)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_clock." + std::to_string(getpid())});
    auto lockfile = miopen::FSLockFile(lockpath);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    EXPECT_TRUE(lockfile.timed_lock(timeout));

    auto timeout2 = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    EXPECT_FALSE(lockfile.timed_lock(timeout2));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, DesynchronizedClocks)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_desync." + std::to_string(getpid())});
    auto lockfile                 = miopen::FSLockFile(lockpath);
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);

    auto future_time = miopen::fs::file_time_type::clock::now() + std::chrono::hours(1);
    miopen::fs::last_write_time(fs_lock_path, future_time);

    EXPECT_TRUE(lockfile.timed_lock(ToPTime(std::chrono::milliseconds(100))));

    lockfile.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, UniqueHandleDeletedDuringLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_deleted." + std::to_string(getpid())});
    auto lockfile      = miopen::FSLockFile(lockpath);
    auto unique_handle = lockfile.get_unique_handle();

    EXPECT_TRUE(lockfile.try_lock());

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    miopen::fs::remove(unique_handle);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    lockfile.unlock();
}

// ========== Shared Lock Tests ==========

TEST(CPU_UnitTestLockFile_NONE, ExclusiveBlocksShared)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_excl_blocks_shared." + std::to_string(getpid())});

    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_TRUE(exclusive_lock.try_lock());

    // Shared lock should fail while exclusive lock held
    miopen::FSLockFile shared_lock(lockpath);
    EXPECT_FALSE(shared_lock.try_lock_shared());

    // Timed shared lock should also timeout
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(shared_lock.timed_lock_shared(ToPTime(std::chrono::milliseconds{50})));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    EXPECT_GE(elapsed.count(), 40);
    EXPECT_LE(elapsed.count(), 100);

    exclusive_lock.unlock();

    // Now shared lock should succeed
    EXPECT_TRUE(shared_lock.try_lock_shared());
    shared_lock.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, SharedBlocksExclusive)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared_blocks_excl." + std::to_string(getpid())});

    miopen::FSLockFile shared_lock(lockpath);
    EXPECT_TRUE(shared_lock.try_lock_shared());

    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_FALSE(exclusive_lock.try_lock());

    shared_lock.unlock_shared();

    EXPECT_TRUE(exclusive_lock.try_lock());
    exclusive_lock.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, MultipleSharedLocksReleaseOrder)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi_release." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    std::vector<std::unique_ptr<miopen::FSLockFile>> locks;

    for(int i = 0; i < 5; ++i)
    {
        auto lock = std::make_unique<miopen::FSLockFile>(lockpath);
        EXPECT_TRUE(lock->try_lock_shared());
        locks.push_back(std::move(lock));
    }

    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    for(int i = 0; i < 4; ++i)
    {
        locks[i]->unlock_shared();
        EXPECT_TRUE(miopen::fs::exists(fs_lock_path))
            << "Directory removed too early after releasing lock " << i;
    }

    locks[4]->unlock_shared();
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, ConcurrentSharedLockCreation)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent_create." + std::to_string(getpid())});

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<miopen::FSLockFile>> locks(10);

    for(int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&, i]() {
            locks[i] = std::make_unique<miopen::FSLockFile>(lockpath);
            if(locks[i]->try_lock_shared())
            {
                success_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    for(auto& t : threads)
        t.join();

    EXPECT_EQ(success_count.load(), 10);

    for(auto& lock : locks)
    {
        if(lock)
            lock->unlock_shared();
    }
}

TEST(CPU_UnitTestLockFile_NONE, AlternatingExclusiveSharedLocks)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_alternating." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    for(int i = 0; i < 10; ++i)
    {
        {
            miopen::FSLockFile excl(lockpath);
            EXPECT_TRUE(excl.try_lock()) << "Exclusive lock failed on iteration " << i;
            EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));
            excl.unlock();
        }

        {
            miopen::FSLockFile shared(lockpath);
            EXPECT_TRUE(shared.try_lock_shared()) << "Shared lock failed on iteration " << i;
            EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));
            shared.unlock_shared();
        }
    }
}

TEST(CPU_UnitTestLockFile_NONE, RapidSharedToExclusiveTransition)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_rapid_transition." + std::to_string(getpid())});

    for(int i = 0; i < 20; ++i)
    {
        miopen::FSLockFile shared_lock(lockpath);
        EXPECT_TRUE(shared_lock.try_lock_shared());
        shared_lock.unlock_shared();

        miopen::FSLockFile excl_lock(lockpath);
        EXPECT_TRUE(excl_lock.try_lock())
            << "Failed to acquire exclusive lock after shared unlock on iteration " << i;
        excl_lock.unlock();
    }
}

TEST(CPU_UnitTestLockFile_NONE, DirectoryToFileTransition)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_dir_to_file." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    {
        miopen::FSLockFile shared(lockpath);
        EXPECT_TRUE(shared.try_lock_shared());
        EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));
        shared.unlock_shared();
    }

    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));

    {
        miopen::FSLockFile excl(lockpath);
        EXPECT_TRUE(excl.try_lock());
        EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
        EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));
        excl.unlock();
    }

    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, StaleSharedLockCleanup)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_stale_excl." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);
    miopen::fs::last_write_time(fs_lock_path, miopen::fs::file_time_type::clock::now());

    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));

    miopen::FSLockFile shared_lock(lockpath);
    EXPECT_FALSE(shared_lock.try_lock_shared());
    EXPECT_TRUE(shared_lock.timed_lock_shared(ToPTime(std::chrono::milliseconds{50})));

    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    shared_lock.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, ExclusiveWithStaleReaders)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_stale_readers." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    std::error_code ec;
    miopen::fs::create_directories(fs_lock_path, ec);

    for(int i = 0; i < 3; ++i)
    {
        miopen::fs::path reader_file = fs_lock_path.string() + "/stale_reader_" + std::to_string(i);
        EXPECT_TRUE(std::ofstream{reader_file});
        miopen::fs::last_write_time(reader_file, miopen::fs::file_time_type::clock::now());
    }

    miopen::FSLockFile excl_lock(lockpath);
    EXPECT_TRUE(excl_lock.timed_lock(ToPTime(std::chrono::milliseconds{50})));

    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));

    excl_lock.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, SharedLockWithExistingReaderFile)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_existing_reader." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    miopen::FSLockFile lock1(lockpath);
    EXPECT_TRUE(lock1.try_lock_shared());

    miopen::FSLockFile lock2(lockpath);
    EXPECT_TRUE(lock2.try_lock_shared());

    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    lock1.unlock_shared();
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));

    lock2.unlock_shared();
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

// ========== Edge Cases ==========

TEST(CPU_UnitTestLockFile_NONE, MultipleLocksIndependent)
{
    auto lockpath1 = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi1." + std::to_string(getpid())});
    auto lockpath2 = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi2." + std::to_string(getpid())});

    miopen::fs::path fs_lock_path1 = lockpath1.string() + ".fslock";
    miopen::fs::path fs_lock_path2 = lockpath2.string() + ".fslock";

    EXPECT_TRUE(std::ofstream{fs_lock_path1});
    EXPECT_TRUE(std::ofstream{fs_lock_path2});
    miopen::fs::permissions(fs_lock_path1, miopen::fs::perms::all);
    miopen::fs::permissions(fs_lock_path2, miopen::fs::perms::all);

    auto now = miopen::fs::file_time_type::clock::now();
    miopen::fs::last_write_time(fs_lock_path1, now);
    miopen::fs::last_write_time(fs_lock_path2, now);

    auto lockfile1 = miopen::FSLockFile(lockpath1);
    auto lockfile2 = miopen::FSLockFile(lockpath2);

    EXPECT_TRUE(lockfile1.timed_lock(ToPTime(std::chrono::milliseconds(100))));
    EXPECT_TRUE(lockfile2.timed_lock(ToPTime(std::chrono::milliseconds(100))));

    lockfile1.unlock();
    lockfile2.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, RefreshThreadTerminationOnFileDelete)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_refresh_term." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    miopen::FSLockFile lock(lockpath);
    EXPECT_TRUE(lock.try_lock_shared());

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::error_code ec;
    for(const auto& entry : miopen::fs::directory_iterator(fs_lock_path, ec))
    {
        miopen::fs::remove(entry.path(), ec);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    lock.unlock_shared();

    EXPECT_NO_THROW(miopen::fs::remove(fs_lock_path, ec));
}

TEST(CPU_UnitTestLockFile_NONE, EmptyDirectoryRaceCondition)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_empty_race." + std::to_string(getpid())});

    std::atomic<bool> shared_started{false};
    std::atomic<bool> exclusive_attempted{false};

    std::thread shared_thread([&]() {
        miopen::FSLockFile lock(lockpath);
        lock.lock_shared();
        shared_started = true;

        while(!exclusive_attempted.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        lock.unlock_shared();
    });

    while(!shared_started.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    miopen::FSLockFile excl(lockpath);
    EXPECT_FALSE(excl.try_lock());
    exclusive_attempted = true;

    shared_thread.join();
}

TEST(CPU_UnitTestLockFile_NONE, ConcurrentExclusiveSharedAttempts)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent_types." + std::to_string(getpid())});

    std::atomic<int> exclusive_acquired{0};
    std::atomic<int> shared_acquired{0};
    std::vector<std::thread> threads;

    for(int i = 0; i < 5; ++i)
    {
        threads.emplace_back([&lockpath, &exclusive_acquired]() {
            miopen::FSLockFile lock(lockpath);
            if(lock.try_lock())
            {
                exclusive_acquired++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.unlock();
            }
        });

        threads.emplace_back([&lockpath, &shared_acquired]() {
            miopen::FSLockFile lock(lockpath);
            if(lock.try_lock_shared())
            {
                shared_acquired++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.unlock_shared();
            }
        });
    }

    for(auto& t : threads)
        t.join();

    EXPECT_GT(exclusive_acquired.load() + shared_acquired.load(), 0);
}

// ========== Production Usage Pattern Tests (std::unique_lock / std::shared_lock) ==========

TEST(CPU_UnitTestLockFile_NONE, StdUniqueLockBasic)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_unique_lock." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    {
        std::unique_lock<miopen::FSLockFile> lock(lockfile, std::defer_lock);
        EXPECT_FALSE(lock.owns_lock());

        lock.lock();
        EXPECT_TRUE(lock.owns_lock());

        miopen::FSLockFile lockfile2(lockpath);
        EXPECT_FALSE(lockfile2.try_lock());
    }

    miopen::FSLockFile lockfile3(lockpath);
    EXPECT_TRUE(lockfile3.try_lock());
    lockfile3.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, StdUniqueLockTimeoutBehavior)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_unique_timeout." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    miopen::FSLockFile lockfile2(lockpath);

    // Success case
    {
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        std::unique_lock<miopen::FSLockFile> lock(lockfile1, timeout);
        EXPECT_TRUE(lock.owns_lock());

        // Failure case - timeout while lock held
        auto timeout2 = std::chrono::steady_clock::now() + std::chrono::milliseconds{50};
        std::unique_lock<miopen::FSLockFile> lock2(lockfile2, timeout2);
        EXPECT_FALSE(lock2.owns_lock());
    }

    // After first lock released, should succeed
    auto timeout3 = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    std::unique_lock<miopen::FSLockFile> lock3(lockfile2, timeout3);
    EXPECT_TRUE(lock3.owns_lock());
}

TEST(CPU_UnitTestLockFile_NONE, StdSharedLockBasic)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared_lock." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    {
        std::shared_lock<miopen::FSLockFile> lock(lockfile, std::defer_lock);
        EXPECT_FALSE(lock.owns_lock());

        lock.lock();
        EXPECT_TRUE(lock.owns_lock());

        miopen::FSLockFile lockfile2(lockpath);
        EXPECT_TRUE(lockfile2.try_lock_shared());
        lockfile2.unlock_shared();

        miopen::FSLockFile lockfile3(lockpath);
        EXPECT_FALSE(lockfile3.try_lock());
    }

    miopen::FSLockFile lockfile4(lockpath);
    EXPECT_TRUE(lockfile4.try_lock());
    lockfile4.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, StdSharedLockWithTimeout)
{
    auto lockpath = miopen::LockFilePath(miopen::fs::path{
        "/tmp/config/miopen/test_shared_timeout_lock." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock(lockfile, timeout);

    EXPECT_TRUE(lock.owns_lock());

    miopen::FSLockFile lockfile2(lockpath);
    EXPECT_TRUE(lockfile2.try_lock_shared());
    lockfile2.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, MultipleSharedLockInstances)
{
    auto lockpath = miopen::LockFilePath(miopen::fs::path{
        "/tmp/config/miopen/test_multi_shared_instances." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    miopen::FSLockFile lockfile2(lockpath);
    miopen::FSLockFile lockfile3(lockpath);

    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock1(lockfile1, std::defer_lock);
    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock2(lockfile2, std::defer_lock);
    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock3(lockfile3, std::defer_lock);

    lock1.lock();
    lock2.lock();
    lock3.lock();

    EXPECT_TRUE(lock1.owns_lock());
    EXPECT_TRUE(lock2.owns_lock());
    EXPECT_TRUE(lock3.owns_lock());

    miopen::FSLockFile excl_lockfile(lockpath);
    EXPECT_FALSE(excl_lockfile.try_lock());
}

TEST(CPU_UnitTestLockFile_NONE, NestedScopeReleasePattern)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_nested_scope." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    {
        auto timeout1 = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        std::unique_lock<miopen::FSLockFile> lock1(lockfile, timeout1);
        EXPECT_TRUE(lock1.owns_lock());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    {
        auto timeout2 = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        std::unique_lock<miopen::FSLockFile> lock2(lockfile, timeout2);
        EXPECT_TRUE(lock2.owns_lock());
    }
}

TEST(CPU_UnitTestLockFile_NONE, ConcurrentDatabaseOperationsPattern)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent_db." + std::to_string(getpid())});

    std::atomic<int> successful_ops{0};
    std::vector<std::thread> threads;

    for(int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&lockpath, &successful_ops]() {
            miopen::FSLockFile lockfile(lockpath);

            for(int j = 0; j < 5; ++j)
            {
                auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
                std::unique_lock<miopen::FSLockFile> lock(lockfile, timeout);

                if(lock.owns_lock())
                {
                    successful_ops++;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        });
    }

    for(auto& t : threads)
        t.join();

    EXPECT_EQ(successful_ops.load(), 50);
}

TEST(CPU_UnitTestLockFile_NONE, LockValidationPattern)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_validation." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    EXPECT_TRUE(lockfile1.try_lock());

    miopen::FSLockFile lockfile2(lockpath);
    auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds{10};
    // cppcheck-suppress localMutex
    std::unique_lock<miopen::FSLockFile> lock(lockfile2, timeout);

    if(!lock)
    {
        EXPECT_FALSE(lock.owns_lock());
    }
    else
    {
        FAIL() << "Lock should have failed to acquire";
    }

    lockfile1.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, MixedUniqueLockSharedLockPattern)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_mixed_locks." + std::to_string(getpid())});

    miopen::FSLockFile write_lockfile(lockpath);
    miopen::FSLockFile read_lockfile(lockpath);

    {
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        std::unique_lock<miopen::FSLockFile> write_lock(write_lockfile, timeout);
        EXPECT_TRUE(write_lock.owns_lock());

        auto read_timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds{10};
        std::shared_lock<miopen::FSLockFile> read_lock(read_lockfile, read_timeout);
        EXPECT_FALSE(read_lock.owns_lock());
    }

    {
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        std::shared_lock<miopen::FSLockFile> read_lock(read_lockfile, timeout);
        EXPECT_TRUE(read_lock.owns_lock());
    }
}

TEST(CPU_UnitTestLockFile_NONE, UniqueLockMovability)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_move." + std::to_string(getpid())});

    auto acquire_lock = [](const miopen::fs::path& path) -> std::unique_lock<miopen::FSLockFile> {
        static miopen::FSLockFile lockfile(path);
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{1};
        return std::unique_lock<miopen::FSLockFile>(lockfile, timeout);
    };

    {
        auto lock = acquire_lock(lockpath);
        EXPECT_TRUE(lock.owns_lock());

        miopen::FSLockFile test_lockfile(lockpath);
        EXPECT_FALSE(test_lockfile.try_lock());
    }

    miopen::FSLockFile test_lockfile2(lockpath);
    EXPECT_TRUE(test_lockfile2.try_lock());
    test_lockfile2.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, TryLockForBothTypes)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_try_lock_for." + std::to_string(getpid())});

    miopen::FSLockFile lockfile1(lockpath);
    miopen::FSLockFile lockfile2(lockpath);
    miopen::FSLockFile lockfile3(lockpath);
    miopen::FSLockFile lockfile4(lockpath);

    // Test unique_lock try_lock_for
    // cppcheck-suppress localMutex
    std::unique_lock<miopen::FSLockFile> lock1(lockfile1, std::defer_lock);
    EXPECT_TRUE(lock1.try_lock_for(std::chrono::milliseconds{100}));

    // cppcheck-suppress localMutex
    std::unique_lock<miopen::FSLockFile> lock2(lockfile2, std::defer_lock);
    EXPECT_FALSE(lock2.try_lock_for(std::chrono::milliseconds{50}));

    lock1.unlock();

    // Test shared_lock try_lock_for
    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock3(lockfile3, std::defer_lock);
    EXPECT_TRUE(lock3.try_lock_for(std::chrono::milliseconds{100}));

    // cppcheck-suppress localMutex
    std::shared_lock<miopen::FSLockFile> lock4(lockfile4, std::defer_lock);
    EXPECT_TRUE(lock4.try_lock_for(std::chrono::milliseconds{100}));

    EXPECT_TRUE(lock3.owns_lock());
    EXPECT_TRUE(lock4.owns_lock());
}
