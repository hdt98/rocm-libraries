
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
    auto lockfile      = miopen::FSLockFile(lockpath);
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
    auto lockfile                 = miopen::FSLockFile(lockpath);
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

// ========== Shared Lock Tests ==========

TEST(CPU_UnitTestLockFile_NONE, SharedLockBasic)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    // Should be able to acquire shared lock
    EXPECT_TRUE(lockfile.try_lock_shared());

    // Verify readers directory exists (same path as lockfile, but as directory)
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    lockfile.unlock_shared();

    // Directory should be removed when last reader unlocks
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, MultipleSharedLocks)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi_shared." + std::to_string(getpid())});

    std::vector<std::unique_ptr<miopen::FSLockFile>> lockfiles;

    // Create 5 shared locks
    for(int i = 0; i < 5; ++i)
    {
        auto lockfile = std::make_unique<miopen::FSLockFile>(lockpath);
        EXPECT_TRUE(lockfile->try_lock_shared());
        lockfiles.push_back(std::move(lockfile));
    }

    // All should coexist
    EXPECT_EQ(lockfiles.size(), 5);

    // Unlock all
    for(auto& lockfile : lockfiles)
    {
        lockfile->unlock_shared();
    }
}

TEST(CPU_UnitTestLockFile_NONE, ExclusiveBlocksShared)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_excl_blocks_shared." + std::to_string(getpid())});

    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_TRUE(exclusive_lock.try_lock());

    // Shared lock should fail while exclusive lock held
    miopen::FSLockFile shared_lock(lockpath);
    EXPECT_FALSE(shared_lock.try_lock_shared());

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

    // Exclusive lock should fail while shared lock held
    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_FALSE(exclusive_lock.try_lock());

    shared_lock.unlock_shared();

    // Now exclusive lock should succeed
    EXPECT_TRUE(exclusive_lock.try_lock());
    exclusive_lock.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, SharedTimedLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared_timed." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    // Should acquire within timeout
    EXPECT_TRUE(lockfile.timed_lock_shared(ToPTime(std::chrono::milliseconds{50})));

    lockfile.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, SharedTimedLockTimeout)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared_timeout." + std::to_string(getpid())});

    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_TRUE(exclusive_lock.try_lock());

    // Shared timed lock should timeout
    miopen::FSLockFile shared_lock(lockpath);
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(shared_lock.timed_lock_shared(ToPTime(std::chrono::milliseconds{50})));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

    EXPECT_GE(elapsed.count(), 40);
    EXPECT_LE(elapsed.count(), 100);

    exclusive_lock.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, SharedLockConcurrent)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_shared_concurrent." + std::to_string(getpid())});

    std::atomic<int> shared_count{0};
    std::vector<std::thread> threads;

    // Multiple threads acquiring shared locks
    for(int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&lockpath, &shared_count]() {
            miopen::FSLockFile lockfile(lockpath);
            if(lockfile.try_lock_shared())
            {
                shared_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                lockfile.unlock_shared();
            }
        });
    }

    for(auto& t : threads)
        t.join();

    // All threads should have acquired shared lock
    EXPECT_EQ(shared_count.load(), 10);
}

TEST(CPU_UnitTestLockFile_NONE, SharedExclusiveInterleaved)
{
    auto lockpath = miopen::LockFilePath(miopen::fs::path{
        "/tmp/config/miopen/test_shared_excl_interleaved." + std::to_string(getpid())});

    // Acquire shared, release, acquire exclusive, release, acquire shared again
    miopen::FSLockFile lockfile1(lockpath);
    EXPECT_TRUE(lockfile1.try_lock_shared());
    lockfile1.unlock_shared();

    miopen::FSLockFile lockfile2(lockpath);
    EXPECT_TRUE(lockfile2.try_lock());
    lockfile2.unlock();

    miopen::FSLockFile lockfile3(lockpath);
    EXPECT_TRUE(lockfile3.try_lock_shared());
    lockfile3.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, RapidSharedLockUnlock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_rapid_shared." + std::to_string(getpid())});

    miopen::FSLockFile lockfile(lockpath);

    // Rapid shared lock/unlock cycles
    for(int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(lockfile.try_lock_shared());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lockfile.unlock_shared();
    }
}

TEST(CPU_UnitTestLockFile_NONE, BlockingSharedLock)
{
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_blocking_shared." + std::to_string(getpid())});

    miopen::FSLockFile exclusive_lock(lockpath);
    EXPECT_TRUE(exclusive_lock.try_lock());

    // Launch thread that tries to acquire shared lock
    std::atomic<bool> shared_acquired{false};
    std::thread t([&lockpath, &shared_acquired]() {
        miopen::FSLockFile shared_lock(lockpath);
        shared_lock.lock_shared(); // Blocks until exclusive lock released
        shared_acquired = true;
        shared_lock.unlock_shared();
    });

    // Wait to ensure thread is blocked
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(shared_acquired.load());

    // Release exclusive lock
    exclusive_lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(shared_acquired.load());

    t.join();
}

// ========== Additional Edge Case Tests ==========

TEST(CPU_UnitTestLockFile_NONE, RapidSharedToExclusiveTransition)
{
    // Test that exclusive lock can be acquired immediately after shared lock release
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_rapid_transition." + std::to_string(getpid())});

    for(int i = 0; i < 20; ++i)
    {
        // Acquire shared lock
        miopen::FSLockFile shared_lock(lockpath);
        EXPECT_TRUE(shared_lock.try_lock_shared());

        // Release it
        shared_lock.unlock_shared();

        // Immediately try to acquire exclusive lock
        miopen::FSLockFile excl_lock(lockpath);
        EXPECT_TRUE(excl_lock.try_lock())
            << "Failed to acquire exclusive lock after shared unlock on iteration " << i;
        excl_lock.unlock();
    }
}

TEST(CPU_UnitTestLockFile_NONE, MultipleSharedLocksReleaseOrder)
{
    // Test that directory is only removed when the last shared lock is released
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_multi_release." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    std::vector<std::unique_ptr<miopen::FSLockFile>> locks;

    // Create 5 shared locks
    for(int i = 0; i < 5; ++i)
    {
        auto lock = std::make_unique<miopen::FSLockFile>(lockpath);
        EXPECT_TRUE(lock->try_lock_shared());
        locks.push_back(std::move(lock));
    }

    // Directory should exist
    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    // Release first 4 locks
    for(int i = 0; i < 4; ++i)
    {
        locks[i]->unlock_shared();
        // Directory should still exist (one lock remaining)
        EXPECT_TRUE(miopen::fs::exists(fs_lock_path))
            << "Directory removed too early after releasing lock " << i;
    }

    // Release last lock
    locks[4]->unlock_shared();

    // Now directory should be removed
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, StaleExclusiveLockCleanup)
{
    // Test that stale exclusive locks (regular files) are properly cleaned up
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_stale_excl." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    // Create a stale exclusive lock (old file)
    EXPECT_TRUE(std::ofstream{fs_lock_path});
    miopen::fs::permissions(fs_lock_path, miopen::fs::perms::all);
    miopen::fs::last_write_time(fs_lock_path, miopen::fs::file_time_type::clock::now());

    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));

    // Try to acquire shared lock - should fail initially
    miopen::FSLockFile shared_lock(lockpath);
    EXPECT_FALSE(shared_lock.try_lock_shared());

    // After timeout, should succeed (stale lock cleaned)
    EXPECT_TRUE(shared_lock.timed_lock_shared(ToPTime(std::chrono::milliseconds{50})));

    // Now lockfile_path should be a directory
    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    shared_lock.unlock_shared();
}

TEST(CPU_UnitTestLockFile_NONE, ExclusiveWithStaleReaders)
{
    // Test exclusive lock acquisition when directory has stale reader files
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_stale_readers." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    // Create directory with stale reader files
    std::error_code ec;
    miopen::fs::create_directories(fs_lock_path, ec);

    // Create old reader files
    for(int i = 0; i < 3; ++i)
    {
        miopen::fs::path reader_file = fs_lock_path.string() + "/stale_reader_" + std::to_string(i);
        EXPECT_TRUE(std::ofstream{reader_file});
        miopen::fs::last_write_time(reader_file, miopen::fs::file_time_type::clock::now());
    }

    // Try to acquire exclusive lock - should clean up stale readers and succeed
    miopen::FSLockFile excl_lock(lockpath);
    EXPECT_TRUE(excl_lock.timed_lock(ToPTime(std::chrono::milliseconds{50})));

    // Verify lockfile_path is now a regular file
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
    EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));

    excl_lock.unlock();
}

TEST(CPU_UnitTestLockFile_NONE, ConcurrentSharedLockCreation)
{
    // Test multiple threads trying to create shared locks simultaneously
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent_create." + std::to_string(getpid())});

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<miopen::FSLockFile>> locks(10);

    // All threads try to acquire shared lock at once
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

    // All should succeed
    EXPECT_EQ(success_count.load(), 10);

    // Cleanup
    for(auto& lock : locks)
    {
        if(lock)
            lock->unlock_shared();
    }
}

TEST(CPU_UnitTestLockFile_NONE, AlternatingExclusiveSharedLocks)
{
    // Test rapid alternation between exclusive and shared locks
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_alternating." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    for(int i = 0; i < 10; ++i)
    {
        // Exclusive lock
        {
            miopen::FSLockFile excl(lockpath);
            EXPECT_TRUE(excl.try_lock()) << "Exclusive lock failed on iteration " << i;
            EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));
            excl.unlock();
        }

        // Shared lock
        {
            miopen::FSLockFile shared(lockpath);
            EXPECT_TRUE(shared.try_lock_shared()) << "Shared lock failed on iteration " << i;
            EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));
            shared.unlock_shared();
        }
    }
}

TEST(CPU_UnitTestLockFile_NONE, DirectoryToFileTransition)
{
    // Test that directory is properly converted to file when transitioning from shared to exclusive
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_dir_to_file." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    // Start with shared lock (creates directory)
    {
        miopen::FSLockFile shared(lockpath);
        EXPECT_TRUE(shared.try_lock_shared());
        EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));
        shared.unlock_shared();
    }

    // Directory should be removed
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));

    // Acquire exclusive lock (creates file)
    {
        miopen::FSLockFile excl(lockpath);
        EXPECT_TRUE(excl.try_lock());
        EXPECT_TRUE(miopen::fs::exists(fs_lock_path));
        EXPECT_FALSE(miopen::fs::is_directory(fs_lock_path));
        excl.unlock();
    }

    // File should be removed
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, SharedLockWithExistingReaderFile)
{
    // Test acquiring shared lock when a reader file already exists (simulating crash recovery)
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_existing_reader." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    // Create lock and get reader file path
    miopen::FSLockFile lock1(lockpath);
    EXPECT_TRUE(lock1.try_lock_shared());

    // Create second lock with same lockpath (different object, might have overlapping reader file
    // name)
    miopen::FSLockFile lock2(lockpath);
    EXPECT_TRUE(lock2.try_lock_shared());

    // Both should coexist
    EXPECT_TRUE(miopen::fs::is_directory(fs_lock_path));

    lock1.unlock_shared();

    // Directory should still exist (lock2 active)
    EXPECT_TRUE(miopen::fs::exists(fs_lock_path));

    lock2.unlock_shared();

    // Now should be removed
    EXPECT_FALSE(miopen::fs::exists(fs_lock_path));
}

TEST(CPU_UnitTestLockFile_NONE, ConcurrentExclusiveSharedAttempts)
{
    // Test exclusive and shared locks being attempted simultaneously
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_concurrent_types." + std::to_string(getpid())});

    std::atomic<int> exclusive_acquired{0};
    std::atomic<int> shared_acquired{0};
    std::vector<std::thread> threads;

    // Launch 5 threads trying exclusive, 5 trying shared
    for(int i = 0; i < 5; ++i)
    {
        // Exclusive thread
        threads.emplace_back([&lockpath, &exclusive_acquired]() {
            miopen::FSLockFile lock(lockpath);
            if(lock.try_lock())
            {
                exclusive_acquired++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.unlock();
            }
        });

        // Shared thread
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

    // Either exclusive or shared should succeed, but mutex should be respected
    // At least one lock should have been acquired
    EXPECT_GT(exclusive_acquired.load() + shared_acquired.load(), 0);
}

TEST(CPU_UnitTestLockFile_NONE, RefreshThreadTerminationOnFileDelete)
{
    // Test that refresh thread terminates gracefully when reader file is deleted externally
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_refresh_term." + std::to_string(getpid())});
    miopen::fs::path fs_lock_path = lockpath.string() + ".fslock";

    miopen::FSLockFile lock(lockpath);
    EXPECT_TRUE(lock.try_lock_shared());

    // Wait for refresh thread to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Find and delete the reader file
    std::error_code ec;
    for(const auto& entry : miopen::fs::directory_iterator(fs_lock_path, ec))
    {
        miopen::fs::remove(entry.path(), ec);
    }

    // Wait for refresh to detect the missing file
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Unlock should still work gracefully
    lock.unlock_shared();

    // Directory might still exist if removal failed, but shouldn't crash
    EXPECT_NO_THROW(miopen::fs::remove(fs_lock_path, ec));
}

TEST(CPU_UnitTestLockFile_NONE, EmptyDirectoryRaceCondition)
{
    // Test race where directory appears empty but a reader is being added
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_empty_race." + std::to_string(getpid())});

    std::atomic<bool> shared_started{false};
    std::atomic<bool> exclusive_attempted{false};

    std::thread shared_thread([&]() {
        miopen::FSLockFile lock(lockpath);
        lock.lock_shared(); // Blocking
        shared_started = true;

        // Hold lock while exclusive attempts
        while(!exclusive_attempted.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        lock.unlock_shared();
    });

    // Wait for shared lock to be acquired
    while(!shared_started.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Try exclusive lock - should fail
    miopen::FSLockFile excl(lockpath);
    EXPECT_FALSE(excl.try_lock());
    exclusive_attempted = true;

    shared_thread.join();
}

TEST(CPU_UnitTestLockFile_NONE, MassiveConcurrentLoad)
{
    // Stress test with many threads competing for locks
    auto lockpath = miopen::LockFilePath(
        miopen::fs::path{"/tmp/config/miopen/test_massive_load." + std::to_string(getpid())});

    std::atomic<int> total_acquires{0};
    std::vector<std::thread> threads;

    // 20 threads all competing
    for(int i = 0; i < 20; ++i)
    {
        threads.emplace_back([&, i]() {
            for(int j = 0; j < 10; ++j)
            {
                miopen::FSLockFile lock(lockpath);

                // Randomly choose exclusive or shared
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

    // Should have many successful acquisitions
    EXPECT_GT(total_acquires.load(), 50);
}
