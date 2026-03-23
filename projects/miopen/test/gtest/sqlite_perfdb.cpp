// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <miopen/conv/problem_description.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/process.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/filesystem.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {
using namespace miopen;

static fs::path& exe_path()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static fs::path exe_path;
    return exe_path;
}

static std::optional<fs::path>& thread_logs_root()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::optional<fs::path> path;
    return path;
}

static bool& full_set()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool full_set = false;
    return full_set;
}

struct ArgsHelper
{
    static constexpr const char* logs_path_arg = "thread-logs-root";
    static constexpr const char* write_arg     = "mp-test-child-write";
    static constexpr const char* id_arg        = "mp-test-child";
    static constexpr const char* path_arg      = "mp-test-child-path";

    // Exact gtest filter for spawning child worker processes.
    // Must match the CPU_SQLitePerfDb_ChildWorker_NONE.ProcessWork test case
    static constexpr const char* child_worker_filter =
        "CPU_SQLitePerfDb_ChildWorker_NONE.ProcessWork";
};

class Random
{
public:
    Random(unsigned int seed = 0) : rng(seed), dist_positive(1) {}

    std::mt19937::result_type Next() { return dist(rng); }
    auto NextNonNegative() { return dist_non_negative(rng); }
    auto NextPositive() { return dist_positive(rng); }

private:
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist;
    std::uniform_int_distribution<> dist_non_negative;
    std::uniform_int_distribution<> dist_positive;
};

static Random& Rnd()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static Random rnd;
    return rnd;
}

struct ProblemData : SQLiteSerializable<ProblemData>
{
    conv::ProblemDescription prob;

    ProblemData() : ProblemData(Rnd()) {}
    ProblemData(Random& rnd)
    {
        const int n_inputs          = rnd.NextPositive();
        const int in_height         = rnd.NextPositive();
        const int in_width          = rnd.NextPositive();
        const int kernel_size_h     = rnd.NextPositive();
        const int kernel_size_w     = rnd.NextPositive();
        const int n_outputs         = rnd.NextPositive();
        const int batch_sz          = rnd.NextPositive();
        const int pad_h             = rnd.NextNonNegative();
        const int pad_w             = rnd.NextNonNegative();
        const int kernel_stride_h   = rnd.NextPositive();
        const int kernel_stride_w   = rnd.NextPositive();
        const int kernel_dilation_h = rnd.NextPositive();
        const int kernel_dilation_w = rnd.NextPositive();
        const int bias              = rnd.Next();

        const TensorDescriptor in        = {miopenFloat, {batch_sz, n_inputs, in_height, in_width}};
        const TensorDescriptor weights   = {miopenFloat, {1, 1, kernel_size_h, kernel_size_w}};
        const TensorDescriptor out       = {miopenFloat, {1, n_outputs, 1, 1}};
        const ConvolutionDescriptor conv = {{pad_h, pad_w},
                                            {kernel_stride_h, kernel_stride_w},
                                            {kernel_dilation_h, kernel_dilation_w}};

        prob = {in, weights, out, conv, conv::Direction::Forward, bias};
    }
    ProblemData(int i)
    {
        i += 1;

        const int n_inputs          = i;
        const int in_height         = i;
        const int in_width          = i;
        const int kernel_size_h     = i;
        const int kernel_size_w     = i;
        const int n_outputs         = i;
        const int batch_sz          = i;
        const int pad_h             = i;
        const int pad_w             = i;
        const int kernel_stride_h   = i;
        const int kernel_stride_w   = i;
        const int kernel_dilation_h = i;
        const int kernel_dilation_w = i;
        const int bias              = i;

        const TensorDescriptor in        = {miopenFloat, {batch_sz, n_inputs, in_height, in_width}};
        const TensorDescriptor weights   = {miopenFloat, {1, 1, kernel_size_h, kernel_size_w}};
        const TensorDescriptor out       = {miopenFloat, {1, n_outputs, 1, 1}};
        const ConvolutionDescriptor conv = {{pad_h, pad_w},
                                            {kernel_stride_h, kernel_stride_w},
                                            {kernel_dilation_h, kernel_dilation_w}};

        prob = {in, weights, out, conv, conv::Direction::Forward, bias};
    }

    static std::string table_name() { return "config"; }
    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        conv::ProblemDescription::Visit(self.prob, f);
    }
};

struct SolverData
{
    int x;
    int y;

    struct NoInit
    {
    };

    SolverData(NoInit) : x(0), y(0) {}
    SolverData(Random& rnd) : x(rnd.Next()), y(rnd.Next()) {}
    SolverData() : x(Rnd().Next()), y(Rnd().Next()) {}
    SolverData(int x_, int y_) : x(x_), y(y_) {}

    template <unsigned int seed>
    static SolverData Seeded()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static Random rnd(seed);
        return {static_cast<int>(rnd.Next()), static_cast<int>(rnd.Next())};
    }

    void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = ',';
        SolverData t(NoInit{});
        std::istringstream ss(s);

        const auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

        if(!success)
            return false;

        *this = t;
        return true;
    }

    bool operator==(const SolverData& other) const { return x == other.x && y == other.y; }

private:
    static bool DeserializeField(std::istream& from, int* ret, char separator)
    {
        std::string part;

        if(!std::getline(from, part, separator))
            return false;

        const auto start = part.c_str();
        char* end;
        const auto value = std::strtol(start, &end, 10);

        if(start == end)
            return false;

        *ret = value;
        return true;
    }
};

std::ostream& operator<<(std::ostream& s, const SolverData& td)
{
    s << "x: " << td.x << ", y: " << td.y;
    return s;
}

class DbTest
{
public:
    DbTest() : temp_file("miopen.tests.perfdb"), db_inst{DbKinds::PerfDb, temp_file, false} {}

    virtual ~DbTest()
    {
        std::error_code ec;
        fs::remove(LockFilePath(temp_file.Path()), ec);
    }

protected:
    TempFile temp_file;
    SQLitePerfDb db_inst;

    static const std::array<std::pair<std::string, SolverData>, 2>& common_data()
    {
        static const std::array<std::pair<std::string, SolverData>, 2> data{{
            {id1(), value1()},
            {id0(), value0()},
        }};

        return data;
    }

    void ResetDb() const { db_inst.sql.Exec("delete from config; delete from perf_db;"); }

    static const ProblemData& key()
    {
        static const ProblemData p(0);
        return p;
    }
    static const SolverData& value0()
    {
        static const SolverData data(3, 4);
        return data;
    }

    static const SolverData& value1()
    {
        static const SolverData data(5, 6);
        return data;
    }

    static const SolverData& value2()
    {
        static const SolverData data(7, 8);
        return data;
    }

    static const std::string& id0()
    {
        static const std::string id0_ = "Solver0";
        return id0_;
    }
    static const std::string& id1()
    {
        static const std::string id1_ = "Solver1";
        return id1_;
    }
    static const std::string& id2()
    {
        static const std::string id2_ = "Solver2";
        return id2_;
    }
    static const std::string& missing_id()
    {
        static const std::string missing_id_ = "UnknownSolver";
        return missing_id_;
    }

    template <class TDb, class TKey, class TValue, size_t count>
    static void ValidateSingleEntry(TKey& key,
                                    const std::array<std::pair<std::string, TValue>, count> values,
                                    TDb db)
    {
        auto record = db.FindRecord(key);

        ASSERT_TRUE(record);

        for(const auto& id_value : values)
        {
            TValue read;
            ASSERT_TRUE(record->GetValues(id_value.first, read));
            ASSERT_EQ(id_value.second, read);
        }
    }

    template <class TKey, class TValue, size_t count>
    static void RawWrite(const fs::path& db_path,
                         const TKey& key,
                         const std::array<std::pair<std::string, TValue>, count> values)
    {
        SQLitePerfDb tmp_inst(DbKinds::PerfDb, db_path, false);
        for(const auto& id_values : values)
        {
            tmp_inst.UpdateUnsafe(key, id_values.first, id_values.second);
        }
    }

    static fs::path LockFilePath(const fs::path& db_path) { return db_path + ".test.lock"; }
};

class SchemaTest : public DbTest
{
public:
    void Run() const
    {
        SQLite::result_type res = db_inst.sql.Exec(
            // clang-format off
                "SELECT name, sql "
                "FROM sqlite_master "
                "WHERE type='table' "
                "AND name = 'config';"
            // clang-format on
        );
        ASSERT_EQ(res.size(), 1u);
        res = db_inst.sql.Exec(
            // clang-format off
                "SELECT name, sql "
                "FROM sqlite_master "
                "WHERE type='table' "
                "AND name = 'perf_db';"
            // clang-format on
        );
        ASSERT_EQ(res.size(), 1u);
    }
};

class DbFindTest : public DbTest
{
public:
    void Run()
    {
        ResetDb();

        const ProblemData p;
        db_inst.InsertConfig(p);

        auto no_rec = db_inst.FindRecord(p);
        ASSERT_FALSE(no_rec);

        auto id = db_inst.GetConfigIDs(p);
        const SolverData sol;
        std::ostringstream ss;
        sol.Serialize(ss);
        db_inst.sql.Exec(
            // clang-format off
            "INSERT OR IGNORE INTO perf_db(config, solver, params) "
            "VALUES( " +
            id + ", '" + id0() + "', '" + ss.str() + "');");
        // clang-format on

        auto sol_res = db_inst.FindRecord(p);
        ASSERT_TRUE(sol_res);
    }
};

class DbOperationsTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Testing different db operations db...");

        ProblemData p;
        const SolverData to_be_rewritten(7, 8);

        {
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

            ASSERT_TRUE(db.Update(p, id0(), to_be_rewritten));
            ASSERT_TRUE(db.Update(p, id1(), to_be_rewritten));

            // Rewritting existing value with other.
            ASSERT_TRUE(db.Update(p, id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            ASSERT_TRUE(db.Update(p, id1(), value1()));
        }

        {
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

            // Rewriting existing value to store it to file.
            ASSERT_TRUE(db.Update(p, id0(), value0()));
        }

        {
            SolverData read0, read1, read_missing;
            const auto read_missing_cmp(read_missing);
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

            // Loading by id not present in record should execute well but return false as nothing
            // was read.
            ASSERT_FALSE(db.Load(p, missing_id(), read_missing));

            // In such case value should not be changed.
            ASSERT_EQ(read_missing, read_missing_cmp);

            ASSERT_TRUE(db.Load(p, id0(), read0));
            ASSERT_TRUE(db.Load(p, id1(), read1));

            ASSERT_EQ(read0, value0());
            ASSERT_EQ(read1, value1());

            ASSERT_TRUE(db.Remove(p, id0()));

            read0 = read_missing_cmp;

            ASSERT_FALSE(db.Load(p, id0(), read0));
            ASSERT_TRUE(db.Load(p, id1(), read1));

            ASSERT_EQ(read0, read_missing_cmp);
            ASSERT_EQ(read1, value1());
        }

        {
            SolverData read0, read1;
            const auto read_missing_cmp(read0);
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);

            ASSERT_FALSE(db.Load(p, id0(), read0));
            ASSERT_TRUE(db.Load(p, id1(), read1));

            ASSERT_EQ(read0, read_missing_cmp);
            ASSERT_EQ(read1, value1());
        }
    }
};

class DbParallelTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing db for using two objects targeting one file existing in one "
                          "scope...");

        ProblemData p;

        SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
        ASSERT_TRUE(db.Update(p, id0(), value0()));

        {
            SQLitePerfDb db0(DbKinds::PerfDb, temp_file, false);
            SQLitePerfDb db1(DbKinds::PerfDb, temp_file, false);

            auto r0 = db0.FindRecord(p);
            auto r1 = db1.FindRecord(p);

            ASSERT_TRUE(r0);
            ASSERT_TRUE(r1);

            ASSERT_TRUE(r0->SetValues(id1(), value1()));
            ASSERT_TRUE(r1->SetValues(id2(), value2()));
        }

        const std::array<std::pair<std::string, SolverData>, 3> data{{
            {id0(), value0()},
            {id1(), value1()},
            {id2(), value2()},
        }};
        ASSERT_TRUE(db.Update(p, id1(), value1()));
        ASSERT_TRUE(db.Update(p, id2(), value2()));

        ValidateSingleEntry(p, data, SQLitePerfDb(DbKinds::PerfDb, temp_file, false));
    }
};

class DBMultiThreadedTestWork
{
public:
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int threads_count;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int common_part_size;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int unique_part_size;
    static constexpr unsigned int ids_per_key      = 16;
    static constexpr unsigned int common_part_seed = 435345;

    static const std::vector<SolverData>& common_part()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        static const auto& ref = common_part_init();
        return ref;
    }

    static void Initialize() { (void)common_part(); }

    template <class TDbConstructor>
    static void
    WorkItem(unsigned int id, const TDbConstructor& db_constructor, const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [id, &db_constructor]() {
            CommonPart(db_constructor);
            UniquePart(id, db_constructor);
        });
    }

    template <class TDbConstructor>
    static void ReadWorkItem(unsigned int id,
                             const TDbConstructor& db_constructor,
                             const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [&db_constructor]() { ReadCommonPart(db_constructor); });
    }

    template <class TDbConstructor>
    static void FillForReading(const TDbConstructor& db_constructor)
    {
        CommonPartSection(0u, common_part_size, db_constructor);
    }

    template <class TDbConstructor>
    static void ValidateCommonPart(const TDbConstructor& db_constructor)
    {
        auto db       = db_constructor();
        const auto cp = common_part();

        for(unsigned int i = 0u; i < common_part_size; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            SolverData read(SolverData::NoInit{});

            ASSERT_TRUE(db.Load(p, id, read));
            ASSERT_EQ(read, data);
        }
    }

private:
    template <class TWorker>
    static void RedirectLogs(unsigned int id, const std::string& log_postfix, const TWorker& worker)
    {
        std::ofstream log;
        std::ofstream log_err;
        std::streambuf *cout_buf = nullptr, *cerr_buf = nullptr;

        if(thread_logs_root().has_value())
        {
            // NOLINTBEGIN (bugprone-unchecked-optional-access)
            const auto out_path = thread_logs_root().value() /
                                  ("thread-" + std::to_string(id) + "_" + log_postfix + ".log");
            const auto err_path = thread_logs_root().value() /
                                  ("thread-" + std::to_string(id) + "_" + log_postfix + "-err.log");
            // NOLINTEND (bugprone-unchecked-optional-access)

            fs::remove(out_path);
            fs::remove(err_path);

            log.open(out_path);
            log_err.open(err_path);
            cout_buf = std::cout.rdbuf();
            cerr_buf = std::cerr.rdbuf();
            std::cout.rdbuf(log.rdbuf());
            std::cerr.rdbuf(log_err.rdbuf());
        }

        worker();

        if(thread_logs_root().has_value())
        {
            std::cout.rdbuf(cout_buf);
            std::cerr.rdbuf(cerr_buf);
        }
    }

    template <class TDbConstructor>
    static void ReadCommonPart(const TDbConstructor& db_constructor)
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with common db instance.");
        {
            ReadCommonPartSection(0u, common_part_size / 2, db_constructor);
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with separate db instances.");
        ReadCommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    ReadCommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(unsigned int i = start; i < end; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            SolverData read(SolverData::NoInit{});

            ASSERT_TRUE(db_getter().Load(p, id, read));
            ASSERT_EQ(read, data);
        }
    }

    template <class TDbConstructor>
    static void CommonPart(const TDbConstructor& db_constructor)
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with common db instance.");
        {
            CommonPartSection(0u, common_part_size / 2, db_constructor);
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with separate db instances.");
        CommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void CommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(unsigned int i = start; i < end; i++)
        {
            ProblemData p(static_cast<int>(i / ids_per_key));
            const auto id   = i % ids_per_key;
            const auto data = cp[i];

            db_getter().Update(p, std::to_string(id), data);
        }
    }

    template <class TDbConstructor>
    static void UniquePart(unsigned int id, const TDbConstructor& db_constructor)
    {
        Random rnd(123123 + id);

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Unique part. Section with common db instance.");
        {
            UniquePartSection(rnd, 0, unique_part_size / 2, db_constructor);
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Unique part. Section with separate db instances.");
        UniquePartSection(rnd, unique_part_size / 2, unique_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    UniquePartSection(Random& rnd, unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        for(auto i = start; i < end; i++)
        {
            auto id = LimitedRandom(rnd, ids_per_key + 1);
            SolverData data(rnd);
            ProblemData p;

            db_getter().Update(p, std::to_string(id), data);
        }
    }

    static std::mt19937::result_type LimitedRandom(Random& rnd, std::mt19937::result_type min)
    {
        std::mt19937::result_type key;

        do
            key = rnd.Next();
        while(key < min);

        return key;
    }

    static const std::vector<SolverData>& common_part_init()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::vector<SolverData> data(common_part_size, SolverData::NoInit{});

        for(auto i = 0u; i < common_part_size; i++)
            data[i] = SolverData::Seeded<common_part_seed>();

        return data;
    }
};

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::threads_count = 16;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::common_part_size = 16;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::unique_part_size = 16;

class DbMultiThreadedTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multithreaded write access...");

        ResetDb();
        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }
};

class DbMultiThreadedReadTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multithreaded read access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();
    }
};

class DbMultiProcessTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multiprocess write access...");

        ResetDb();
        std::vector<ProcessAsync> children{};
        const auto lock_file_path = LockFilePath(temp_file);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test processes...");
        {
            auto& file_lock = miopen::LockFile::Get(lock_file_path);
            std::shared_lock<miopen::LockFile> lock(file_lock);

            auto id = 0;

            // clang-format off
            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; ++i)
            {
                auto args =
                    std::string{"--reset-sharding"
                    " --gtest_filter="} + ArgsHelper::child_worker_filter +
                    " --" + ArgsHelper::write_arg +
                    " --" + ArgsHelper::id_arg + " " + std::to_string(id++) +
                    " --" + ArgsHelper::path_arg + " " + temp_file.Path();

                if(thread_logs_root().has_value())
                {
                    // NOLINTNEXTLINE (bugprone-unchecked-optional-access)
                    args += std::string{" --"} + ArgsHelper::logs_path_arg + " " + thread_logs_root().value();
                }

                if(full_set())
                    args += " --all";

                children.emplace_back(exe_path(), args);
            }
            // clang-format on
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test processes...");
        for(auto&& child : children)
        {
            ASSERT_EQ(child.Wait(), 0);
        }

        {
            std::error_code ec;
            fs::remove(lock_file_path, ec);
        }

        const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }

    static void WorkItem(unsigned int id, const std::string& db_path, bool write)
    {
        {
            auto& file_lock = miopen::LockFile::Get(LockFilePath(db_path));
            std::lock_guard<miopen::LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return SQLitePerfDb(DbKinds::PerfDb, db_path, false); };

        if(write)
            DBMultiThreadedTestWork::WorkItem(id, c, "mp");
        else
            DBMultiThreadedTestWork::ReadWorkItem(id, c, "mp");
    }
};

class DbMultiProcessReadTest : public DbTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multiprocess read access...");

        std::vector<ProcessAsync> children{};
        const auto lock_file_path = LockFilePath(temp_file);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]() { return SQLitePerfDb(DbKinds::PerfDb, temp_file, false); };
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test processes...");
        {
            auto& file_lock = miopen::LockFile::Get(lock_file_path);
            std::shared_lock<miopen::LockFile> lock(file_lock);

            auto id = 0;

            // clang-format off
            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; ++i)
            {
                // Note: no --write flag for read test
                auto args =
                    std::string{"--reset-sharding"
                    " --gtest_filter="} + ArgsHelper::child_worker_filter +
                    " --" + ArgsHelper::id_arg + " " + std::to_string(id++) +
                    " --" + ArgsHelper::path_arg + " " + temp_file.Path();

                if(thread_logs_root().has_value())
                {
                    // NOLINTNEXTLINE (bugprone-unchecked-optional-access)
                    args += std::string{" --"} + ArgsHelper::logs_path_arg + " " + thread_logs_root().value();
                }

                if(full_set())
                    args += " --all";

                MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", exe_path() + " " + args);
                children.emplace_back(exe_path(), args);
            }
            // clang-format on
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test processes...");
        for(auto&& child : children)
        {
            ASSERT_EQ(child.Wait(), 0);
        }

        {
            std::error_code ec;
            fs::remove(lock_file_path, ec);
        }
    }
};

class DbMultiFileTest : public DbTest
{
protected:
    const fs::path user_db_path = temp_file.Path() + ".user";

    void ResetDb() const { DbTest::ResetDb(); }
};

template <bool merge_records>
class DbMultiFileReadTest : public DbMultiFileTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Running multifile read test" << (merge_records ? " with merge" : "")
                                                        << "...");

        ResetDb();
        MergedAndMissing();

        ResetDb();
        ReadUser();

        ResetDb();
        ReadInstalled();

        ResetDb();
        ReadConflict();
    }

private:
    static const std::array<std::pair<std::string, SolverData>, 1>& single_item_data()
    {
        static const std::array<std::pair<std::string, SolverData>, 1> data{{{id0(), value2()}}};

        return data;
    }

    void MergedAndMissing() const
    {
        RawWrite(temp_file, key(), common_data());
        RawWrite(user_db_path, key(), single_item_data());

        static const std::array<std::pair<std::string, SolverData>, 2> merged_data{{
            {id1(), value1()},
            {id0(), value2()},
        }};

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records> db(
            DbKinds::PerfDb, temp_file, user_db_path);
        if(merge_records)
            ValidateSingleEntry(key(), merged_data, std::move(db));
        else
            ValidateSingleEntry(key(), single_item_data(), std::move(db));

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records> db1(
            DbKinds::PerfDb, temp_file, user_db_path);
        ProblemData p;
        auto record1 = db1.FindRecord(p);
        ASSERT_FALSE(record1);
    }

    void ReadUser() const
    {
        RawWrite(user_db_path, key(), single_item_data());
        ValidateSingleEntry(key(),
                            single_item_data(),
                            MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records>(
                                DbKinds::PerfDb, temp_file, user_db_path));
    }

    void ReadInstalled() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ValidateSingleEntry(key(),
                            single_item_data(),
                            MultiFileDb<SQLitePerfDb, SQLitePerfDb, merge_records>(
                                DbKinds::PerfDb, temp_file, user_db_path));
    }

    void ReadConflict() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ReadUser();
    }
};

class DbMultiFileWriteTest : public DbMultiFileTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Running multifile write test...");

        ResetDb();

        {
            MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(
                DbKinds::PerfDb, temp_file, user_db_path);
            ASSERT_TRUE(db.StoreRecord(key(), id0(), value0()));
            ASSERT_TRUE(db.Update(key(), id1(), value1()));
        }
        ASSERT_FALSE(SQLitePerfDb(DbKinds::PerfDb, temp_file, false).FindRecord(key()));
        ASSERT_TRUE(SQLitePerfDb(DbKinds::PerfDb, user_db_path, false).FindRecord(key()));

        ValidateSingleEntry(key(),
                            common_data(),
                            MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(
                                DbKinds::PerfDb, temp_file, user_db_path));
    }
};

class DbMultiFileOperationsTest : public DbMultiFileTest
{
public:
    void Run() const
    {
        ResetDb();
        PrepareDb();
        UpdateTest();
        LoadTest();
        RemoveTest();
        RemoveRecordTest();
    }

    void PrepareDb() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Running multifile operations test...");

        {
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
            ASSERT_TRUE(db.StoreRecord(key(), id0(), value0()));
            ASSERT_TRUE(db.Update(key(), id1(), value2()));
        }
    }

    void UpdateTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Update test...");

        {
            MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(
                DbKinds::PerfDb, temp_file, user_db_path);
            ASSERT_TRUE(db.Update(key(), id1(), value1()));
        }

        {
            SQLitePerfDb db(DbKinds::PerfDb, user_db_path, false);
            SolverData read(SolverData::NoInit{});
            ASSERT_FALSE(db.Load(key(), id0(), read));
            ASSERT_TRUE(db.Load(key(), id1(), read));
            ASSERT_EQ(read, value1());
        }

        {
            SQLitePerfDb db(DbKinds::PerfDb, temp_file, false);
            ValidateData(db, value2());
        }
    }

    void LoadTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Load test...");

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        ValidateData(db, value1());
    }

    void RemoveTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Remove test...");

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        ASSERT_TRUE(db.Remove(key(), id0()));
        ASSERT_TRUE(db.Remove(key(), id1()));

        ValidateData(db, value2());
    }

    void RemoveRecordTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Remove record test...");

        MultiFileDb<SQLitePerfDb, SQLitePerfDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        ASSERT_TRUE(db.Update(key(), id1(), value1()));
        ASSERT_TRUE(db.Remove(key(), id1()));

        ValidateData(db, value2());
    }

    template <class TDb>
    void ValidateData(TDb& db, const SolverData& id1Value) const
    {
        SolverData read(SolverData::NoInit{});
        ASSERT_TRUE(db.Load(key(), id0(), read));
        ASSERT_EQ(read, value0());
        ASSERT_TRUE(db.Load(key(), id1(), read));
        ASSERT_EQ(read, id1Value);
    }
};

class DbMultiFileMultiThreadedReadTest : public DbMultiFileTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multifile multithreaded read access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]() {
            return MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(
                DbKinds::PerfDb, temp_file, user_db_path);
        };
        ResetDb();
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();
    }
};

class DbMultiFileMultiThreadedTest : public DbMultiFileTest
{
public:
    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing db for multifile multithreaded write access...");

        ResetDb();
        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const auto c = [this]() {
            return MultiFileDb<SQLitePerfDb, SQLitePerfDb, true>(
                DbKinds::PerfDb, temp_file, user_db_path);
        };

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }
};

std::string GetArg(const std::string& name)
{
    const auto& args  = testing::internal::GetArgvs();
    const auto target = "--" + name;

    for(size_t i = 1; i + 1 < args.size(); ++i)
    {
        if(args[i] == target)
            return args[i + 1];
    }
    return "";
}

bool HasFlag(const std::string& name)
{
    const auto& args  = testing::internal::GetArgvs();
    const auto target = "--" + name;

    for(size_t i = 1; i < args.size(); ++i)
    {
        if(args[i] == target)
            return true;
    }
    return false;
}

bool IsChildProcessMode() { return !GetArg(ArgsHelper::id_arg).empty(); }

class CPU_SQLitePerfDb_ChildWorker_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!IsChildProcessMode())
        {
            GTEST_SKIP() << "Not in child process mode (no --" << ArgsHelper::id_arg << " arg)";
        }
    }
};

// cppcheck false positive: TEST_F inside anonymous namespace with complex fixture definitions.
// Known issue: https://trac.cppcheck.net/ticket/9160 (wontfix).
// Same workaround used in perfdb.cpp and conv_ai_3d_heuristics.cpp.
// cppcheck-suppress syntaxError
TEST_F(CPU_SQLitePerfDb_ChildWorker_NONE, ProcessWork)
{
    std::string logs_root = GetArg(ArgsHelper::logs_path_arg);
    if(!logs_root.empty())
        thread_logs_root() = logs_root;

    if(HasFlag("all"))
    {
        full_set()                                = true;
        DBMultiThreadedTestWork::threads_count    = 16;
        DBMultiThreadedTestWork::common_part_size = 32;
        DBMultiThreadedTestWork::unique_part_size = 32;
    }

    std::string db_path = GetArg(ArgsHelper::path_arg);
    int id              = std::stoi(GetArg(ArgsHelper::id_arg));
    bool write          = HasFlag(ArgsHelper::write_arg);

    DbMultiProcessTest::WorkItem(id, db_path, write);
}

class CPU_SQLitePerfDb_NONE : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        // Save exe path for spawning child processes
        const auto& args = testing::internal::GetArgvs();
        if(!args.empty())
        {
            exe_path() = fs::absolute(args[0]);
#ifdef _WIN32
            // CTest may invoke without .exe extension, but CreateProcess
            // with explicit lpApplicationName requires it
            if(!exe_path().has_extension())
                exe_path().replace_extension(".exe");
#endif
        }
    }
};

TEST_F(CPU_SQLitePerfDb_NONE, SQLitePerfDb_AllTests)
{
    SchemaTest().Run();
    DbFindTest().Run();
    DbOperationsTest().Run();
    DbParallelTest().Run();
    DbMultiThreadedTest().Run();
    DbMultiThreadedReadTest().Run();
    DbMultiProcessReadTest().Run();
    DbMultiProcessTest().Run();
}

TEST_F(CPU_SQLitePerfDb_NONE, MultiFileDb_AllTests)
{
#if !MIOPEN_DISABLE_USERDB
    DbMultiFileReadTest<true>().Run();
    DbMultiFileReadTest<false>().Run();
    DbMultiFileWriteTest().Run();
    DbMultiFileOperationsTest().Run();
    DbMultiFileMultiThreadedReadTest().Run();
    DbMultiFileMultiThreadedTest().Run();
#endif
}

} // anonymous namespace
