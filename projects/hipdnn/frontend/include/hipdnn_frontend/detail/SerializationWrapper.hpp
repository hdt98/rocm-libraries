#include <hipdnn_backend.h>
#include <hipdnn_frontend/Types.hpp>
#include <tuple>

namespace hipdnn_frontend::detail
{
template <hipdnnBackendAttributeName_t K, class V>
struct KeyValue
{
    static constexpr hipdnnBackendAttributeName_t KEY = K;
    using Value = V;

    constexpr hipdnnBackendAttributeName_t key()
    {
        return K;
    }

    Value value;
};

template <hipdnnBackendAttributeName_t A,
          hipdnnBackendAttributeName_t Head,
          hipdnnBackendAttributeName_t... Tail>
constexpr size_t getIndexOf()
{
    if constexpr(A == Head)
    {
        return 0;
    }
    else
    {
        return 1 + getIndexOf<A, Tail...>();
    }
}

constexpr size_t TEST = getIndexOf<HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                   HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                   HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                   HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS>();

template <class... Ts>
class AttributeMap
{
    std::tuple<typename Ts::Value...> _values;

public:
    template <hipdnnBackendAttributeName_t A>
    constexpr auto& at()
    {
        constexpr size_t IDX = getIndexOf<A, Ts::KEY...>();
        return std::get<IDX>(_values);
    }
};

using ForwardConvAttributes
    = AttributeMap<KeyValue<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, int64_t>,
                   KeyValue<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, int64_t>,
                   KeyValue<HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, std::vector<int64_t>>,
                   KeyValue<HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, std::vector<int64_t>>,
                   KeyValue<HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, std::vector<int64_t>>,
                   KeyValue<HIPDNN_ATTR_CONVOLUTION_DILATIONS, std::vector<int64_t>>,
                   KeyValue<HIPDNN_ATTR_CONVOLUTION_CONV_MODE, std::vector<ConvolutionMode>>>;

}
