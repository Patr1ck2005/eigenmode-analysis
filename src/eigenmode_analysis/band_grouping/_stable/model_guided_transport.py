"""Reusable Re model groups with complete transport and explicit input arms.

Uses existing G2 extraction/continuation unchanged. This is an integration
candidate, not a new model family or per-case selection of winning outputs.
Unsupported models retain the same fixed transport fallback for every case.
"""
from .discriminant_transport import reconstruct as reconstruct_transport
from .scalar_model_grouping import reconstruct_model_grouping


def reconstruct(c,f,*,channel='real'):
    fallback,transport_info=reconstruct_transport(c,f,channel=channel)
    mesh,model_info=reconstruct_model_grouping(c,f,fallback=fallback)
    return mesh,{'version':'model-guided-complete-transport-v1','channel':channel,
        'classification_succeeded':True,'status':'conditional_model_and_transport_output',
        'transport':transport_info,'model_grouping':model_info,
        'input_policy':'Existing G2 models use Re only; measured Im is used only in the explicitly selected transport arm.',
        'generated_measurements':0,'stage_qualified':False,'production_default':False,
        'limitations':['Model fitting is conditional; no new truth or global identification proof.',
                       'Global band IDs do not preserve every complex monodromy association.']}
