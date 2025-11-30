from sentinelai.models.Swin_UNETR.data_processing.flaw_sizing_methods.maskSizing_v5_MADmedianCScan import (
    adjust_mask,
)
from sentinelai.plugins.inference_postprocess_hooks.inference_postprocess_plugins.flaw_sizing_base import (
    FlawSizingBase,
)

from ..postprocessing_plugin_manager import (
    postprocess_plugin_registry,
    register_postprocess_plugin,
)


@register_postprocess_plugin(postprocess_plugin_registry)
class FlawSizingv5_MADmedianCScan(FlawSizingBase):
    def __init__(self):
        super().__init__(
            mask_adjustment_method=adjust_mask,
            inference_id="flaw_sizing_v5_MADmedianCScan",
            csv_output="indication_table_{status_info['gr']}_v5_MADmedianCScan",
        )
