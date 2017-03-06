py_binary(
    name = "process_medical_data",
    srcs = ["process_medical_data.py"],
    data = ["//utils:w2v.so"],
    imports = ["utils"],
)

py_binary(
    name = "prepare_data",
    srcs = ["prepare_data.py"],
    data = ["//utils:w2v.so"],
    imports = ["utils"],
)

py_binary(
    name = "sentence_clfier",
    srcs = ["sentence_clfier.py"],
    deps = [
        ":prepare_data",
    ],
)

py_binary(
    name = "prepare_ner_data",
    srcs = ["prepare_ner_data.py"],
    data = ["//utils:w2v.so"],
    imports = ["utils"],
)

