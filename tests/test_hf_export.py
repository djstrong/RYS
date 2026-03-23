from hf_export.common import build_exported_config, build_tensor_name_mapping


def test_build_tensor_name_mapping_duplicates_decoder_layers() -> None:
    weight_map = {
        "model.layers.0.a": "model-00001.safetensors",
        "model.layers.0.b": "model-00001.safetensors",
        "model.layers.1.a": "model-00001.safetensors",
        "model.layers.1.b": "model-00001.safetensors",
        "model.layers.2.a": "model-00002.safetensors",
        "model.layers.2.b": "model-00002.safetensors",
        "model.norm.weight": "model-00002.safetensors",
    }
    mapping = build_tensor_name_mapping(
        weight_map=weight_map,
        text_layer_prefix="model.layers.",
        layer_indices=(0, 1, 1, 2),
    )
    assert mapping["model.layers.0.a"] == "model.layers.0.a"
    assert mapping["model.layers.2.a"] == "model.layers.1.a"
    assert mapping["model.layers.2.b"] == "model.layers.1.b"
    assert mapping["model.layers.3.a"] == "model.layers.2.a"
    assert mapping["model.norm.weight"] == "model.norm.weight"


def test_build_exported_config_updates_text_config_metadata() -> None:
    base_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["a", "b", "c", "d"],
        },
    }
    exported = build_exported_config(
        base_config,
        layer_indices=(0, 1, 1, 2, 3),
        source_num_layers=4,
        source_repo_id="example/base-model",
        spec_text="blocks:1,2",
        text_layer_prefix="model.layers.",
    )
    assert exported["text_config"]["num_hidden_layers"] == 5
    assert exported["text_config"]["layer_types"] == ["a", "b", "b", "c", "d"]
    assert exported["rys_relayer"]["layer_indices"] == [0, 1, 1, 2, 3]


def test_build_exported_config_remaps_modules_to_not_convert() -> None:
    base_config = {
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        },
        "quantization_config": {
            "modules_to_not_convert": [
                "lm_head",
                "model.layers.0.linear_attn.in_proj_a",
                "model.layers.1.linear_attn.in_proj_a",
                "model.layers.3.linear_attn.in_proj_b",
            ]
        },
    }
    exported = build_exported_config(
        base_config,
        layer_indices=(0, 1, 1, 2, 3),
        source_num_layers=4,
        source_repo_id="example/base-model",
        spec_text="layers:0,1,1,2,3",
        text_layer_prefix="model.layers.",
    )
    assert exported["quantization_config"]["modules_to_not_convert"] == [
        "lm_head",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.1.linear_attn.in_proj_a",
        "model.layers.2.linear_attn.in_proj_a",
        "model.layers.4.linear_attn.in_proj_b",
    ]
