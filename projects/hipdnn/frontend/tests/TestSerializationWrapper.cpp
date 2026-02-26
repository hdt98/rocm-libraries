#include <hipdnn_frontend/detail/SerializationWrapper.hpp>

using namespace hipdnn_frontend::detail;

void set()
{
    ForwardConvAttributes attrs;

    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X>(inputs[X_TENSOR], tensorDescs);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y>(inputs[X_TENSOR], tensorDescs);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W>(inputs[X_TENSOR], tensorDescs);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA>(alpha);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA>(beta);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X>(inputs[X_TENSOR]);
    attrs.set<HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X>(inputs[X_TENSOR]);
}
