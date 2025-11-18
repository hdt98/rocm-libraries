#include <sstream>
#include <string>

struct HostArgs
{
    std::unordered_map<std::string, std::string> arg_table;

    HostArgs(int argc, char** argv)
    {
        for(int i = 1; i < argc; i += 2)
        {
            assert(i + 1 < argc && "flag does not match any args");
            arg_table.emplace(argv[i], argv[i + 1]);
        }
    }

    template <typename T>
    void ParseArgs(std::string name, T& val)
    {
        if(arg_table.find(name) == arg_table.end())
            return;

        std::stringstream ss(arg_table[name]);
        ss >> val;
        assert(ss.eof() && "failed to convert string");
    }
};

// // MIOpenDriver default setting
// index_t N = 1;
// index_t C = 1;
// index_t H = 32;
// index_t W = 32;

// index_t K = 1;
// index_t R = 3;
// index_t S = 3;

// index_t pad_h = 0;
// index_t pad_w = 0;

// index_t conv_stride_h = 1;
// index_t conv_stride_w = 1;

// index_t dilation_h = 1;
// index_t dilation_w = 1;

// A useful test
ck::index_t G = 1;

ck::index_t N = 2;
ck::index_t C = 5;
ck::index_t H = 16;
ck::index_t W = 32;

ck::index_t K = 4;
ck::index_t R = 3;
ck::index_t S = 3;

ck::index_t pad_h = 1;
ck::index_t pad_w = 1;

ck::index_t conv_stride_h = 1;
ck::index_t conv_stride_w = 1;

ck::index_t dilation_h = 1;
ck::index_t dilation_w = 1;

// don't support modifying these yet
ck::index_t F           = 1;
ck::index_t spatial_dim = 2;
std::string mode        = "conv";

auto ParseHostArgs = [](int argc, char* argv[]) {
    HostArgs hargs(argc, argv);

    hargs.ParseArgs("-g", G);

    hargs.ParseArgs("-n", N);
    hargs.ParseArgs("-c", C);
    hargs.ParseArgs("--in_h", H);
    hargs.ParseArgs("--in_w", W);

    hargs.ParseArgs("-k", K);
    hargs.ParseArgs("-y", R);
    hargs.ParseArgs("-x", S);

    hargs.ParseArgs("--pad_h", pad_h);
    hargs.ParseArgs("--pad_w", pad_w);

    hargs.ParseArgs("-u", conv_stride_h);
    hargs.ParseArgs("-v", conv_stride_w);

    hargs.ParseArgs("-l", dilation_h);
    hargs.ParseArgs("-j", dilation_w);

    hargs.ParseArgs("-F", F);
    hargs.ParseArgs("--spatial_dim", spatial_dim);
    hargs.ParseArgs("-m", mode);
};