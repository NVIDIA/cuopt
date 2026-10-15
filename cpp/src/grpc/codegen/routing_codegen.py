# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate RoutingProblem proto and the C++ that copies it.

LP and MIP sections do not use this module. It runs only when
field_registry.yaml contains a routing_problem section.
"""


def _parse(entry):
    from generate_conversions import parse_field

    return parse_field(entry)


def _section(registry):
    return registry.get("routing_problem") or None


def _list(section, key):
    for entry in section.get(key) or []:
        body = _parse(entry)
        body.setdefault("member", body["name"])
        yield body


def _nested_fields(body, key):
    for entry in body.get(key) or []:
        field = _parse(entry)
        field.setdefault("member", field["name"])
        yield field


def _defined_records(section):
    """member_records that introduce a proto message (not a second container)."""
    seen = set()
    for rec in _list(section, "member_records"):
        if rec.get("define_message", True) and rec["message_type"] not in seen:
            seen.add(rec["message_type"])
            yield rec


def _defined_keyed_vectors(section):
    seen = set()
    for rec in _list(section, "keyed_vectors"):
        if rec["message_type"] not in seen:
            seen.add(rec["message_type"])
            yield rec


def _defined_keyed_records(section):
    for rec in _list(section, "keyed_records"):
        yield rec


def _optional_messages(section):
    return list(_list(section, "optional_messages"))


def top_level_array_fields(registry):
    section = _section(registry)
    if not section:
        return []
    return [f for f in _list(section, "arrays") if "array_id" in f]


def routing_array_field_id_pairs(registry):
    section = _section(registry)
    if not section:
        return []
    return [
        (f["array_id"], f"routing_problem.arrays.{f['name']}")
        for f in _list(section, "arrays")
        if "array_id" in f
    ]


def routing_container_groups(registry):
    """(container_field_num, label, [(array_id, name, element_size), ...])."""
    from generate_conversions import _default_element_size

    section = _section(registry)
    if not section:
        return []
    groups = []

    message_arrays = {}
    for rec in _defined_records(section):
        message_arrays[rec["message_type"]] = list(
            _nested_fields(rec, "arrays")
        )
    for rec in _list(section, "member_records"):
        arrays = message_arrays.get(rec["message_type"], [])
        groups.append(
            (
                rec["field_num"],
                f"{rec['message_type']} ({rec['name']})",
                [
                    (f["array_id"], f["name"], _default_element_size(f))
                    for f in arrays
                    if "array_id" in f
                ],
            )
        )

    for rec in _list(section, "keyed_vectors"):
        values = rec["values"]
        groups.append(
            (
                rec["field_num"],
                f"{rec['message_type']} ({rec['name']})",
                [
                    (
                        values["array_id"],
                        values["name"],
                        _default_element_size(values),
                    )
                ],
            )
        )

    for rec in _defined_keyed_records(section):
        groups.append(
            (
                rec["field_num"],
                f"{rec['entry_message']} via {rec['message_type']} ({rec['name']})",
                [
                    (f["array_id"], f["name"], _default_element_size(f))
                    for f in _nested_fields(rec, "arrays")
                    if "array_id" in f
                ],
            )
        )

    for rec in _optional_messages(section):
        groups.append(
            (
                rec["field_num"],
                f"{rec['message_type']} ({rec['name']})",
                [
                    (f["array_id"], f["name"], _default_element_size(f))
                    for f in _nested_fields(rec, "arrays")
                    if "array_id" in f
                ],
            )
        )
    return groups


def validate_routing(registry, errors):
    section = _section(registry)
    if not section:
        return
    parent = []
    for f in _list(section, "scalars"):
        parent.append((f["field_num"], f"scalars.{f['name']}"))
    for f in _list(section, "arrays"):
        parent.append((f["field_num"], f"arrays.{f['name']}"))
    for rec in _list(section, "member_records"):
        parent.append((rec["field_num"], f"member_records.{rec['name']}"))
    for rec in _list(section, "keyed_vectors"):
        parent.append((rec["field_num"], f"keyed_vectors.{rec['name']}"))
    for rec in _list(section, "keyed_records"):
        parent.append((rec["field_num"], f"keyed_records.{rec['name']}"))
    for rec in _optional_messages(section):
        parent.append((rec["field_num"], f"optional_messages.{rec['name']}"))
    _unique("RoutingProblem", parent, errors)

    def inner(label, scalars, arrays):
        pairs = [(f["field_num"], f"scalars.{f['name']}") for f in scalars]
        pairs += [(f["field_num"], f"arrays.{f['name']}") for f in arrays]
        _unique(label, pairs, errors)
        aids = [
            (f["array_id"], f"arrays.{f['name']}")
            for f in arrays
            if "array_id" in f
        ]
        _unique(f"{label}::array_id", aids, errors)

    for rec in _defined_records(section):
        inner(
            rec["message_type"],
            list(_nested_fields(rec, "scalars")),
            list(_nested_fields(rec, "arrays")),
        )
    for rec in _defined_keyed_vectors(section):
        key = rec["key"]
        values = rec["values"]
        inner(
            rec["message_type"],
            [{"name": key["name"], "field_num": key["field_num"]}],
            [values],
        )
    for rec in _defined_keyed_records(section):
        inner(
            rec["entry_message"],
            list(_nested_fields(rec, "scalars")),
            list(_nested_fields(rec, "arrays")),
        )
        # The wrapper message's own tags: key plus the repeated entry field.
        _unique(
            rec["message_type"],
            [
                (rec["key"]["field_num"], f"key.{rec['key']['name']}"),
                (rec["entry_field_num"], f"entry.{rec['entry_field']}"),
            ],
            errors,
        )
    for rec in _optional_messages(section):
        inner(rec["message_type"], [], list(_nested_fields(rec, "arrays")))


def _unique(label, pairs, errors):
    seen = {}
    for num, src in pairs:
        if num in seen:
            errors.append(
                f"{label}: duplicate tag {num} (conflict: {seen[num]!r} vs {src!r})"
            )
        else:
            seen[num] = src


def _proto_scalar_type(ftype):
    return ftype


def _emit_array_id_enum(lines, arrays, indent="  "):
    from generate_conversions import _array_wire_type_comment

    numbered = [f for f in arrays if "array_id" in f]
    if not numbered:
        return
    lines.append(
        f"{indent}// ArrayId is the container-relative ArrayChunk.field_id."
    )
    lines.append(f"{indent}// It is not the protobuf tag of the field.")
    lines.append(f"{indent}enum ArrayId {{")
    for f in sorted(numbered, key=lambda item: item["array_id"]):
        comment = f.get("wire_comment") or _array_wire_type_comment(f)
        name = f["name"].upper()
        lines.append(f"{indent}  {name} = {f['array_id']};  // {comment}")
    lines.append(f"{indent}}}")


def _emit_proto_fields(lines, scalars, arrays, indent="  "):
    fields = []
    for f in scalars:
        fields.append(
            (
                f["field_num"],
                f"  {_proto_scalar_type(f['type'])} {f['name']} = {f['field_num']};",
            )
        )
    for f in arrays:
        fields.append(
            (f["field_num"], f"  {f['type']} {f['name']} = {f['field_num']};")
        )
    for _, line in sorted(fields, key=lambda item: item[0]):
        lines.append(indent[:-2] + line if False else line)


def generate_routing_proto(registry):
    section = _section(registry)
    if not section:
        return ""
    lines = [
        "// AUTO-GENERATED by src/grpc/codegen/generate_conversions.py from field_registry.yaml",
        "// DO NOT EDIT — regenerate with: ./build.sh codegen",
        "// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        'syntax = "proto3";',
        "",
        "package cuopt.remote;",
        "",
        'import "cuopt_remote.proto";  // RequestHeader, ProblemCategory',
        "",
        "// RoutingSolution lives in cuopt_routing_solution.proto so ResultResponse",
        "// can embed it without a circular import.",
        "",
    ]

    for rec in _defined_records(section):
        lines.append(f"message {rec['message_type']} {{")
        _emit_array_id_enum(lines, list(_nested_fields(rec, "arrays")))
        _emit_proto_fields(
            lines,
            list(_nested_fields(rec, "scalars")),
            list(_nested_fields(rec, "arrays")),
        )
        lines.append("}")
        lines.append("")

    for rec in _defined_keyed_records(section):
        lines.append(f"message {rec['entry_message']} {{")
        _emit_array_id_enum(lines, list(_nested_fields(rec, "arrays")))
        _emit_proto_fields(
            lines,
            list(_nested_fields(rec, "scalars")),
            list(_nested_fields(rec, "arrays")),
        )
        lines.append("}")
        lines.append("")
        key = rec["key"]
        lines.append(f"message {rec['message_type']} {{")
        lines.append(f"  {key['type']} {key['name']} = {key['field_num']};")
        lines.append(
            f"  repeated {rec['entry_message']} {rec['entry_field']} = {rec['entry_field_num']};"
        )
        lines.append("}")
        lines.append("")

    for rec in _defined_keyed_vectors(section):
        key = rec["key"]
        values = rec["values"]
        lines.append(f"message {rec['message_type']} {{")
        _emit_array_id_enum(lines, [values])
        lines.append(f"  {key['type']} {key['name']} = {key['field_num']};")
        lines.append(
            f"  {values['type']} {values['name']} = {values['field_num']};"
        )
        lines.append("}")
        lines.append("")

    for rec in _optional_messages(section):
        lines.append(f"message {rec['message_type']} {{")
        _emit_array_id_enum(lines, list(_nested_fields(rec, "arrays")))
        _emit_proto_fields(lines, [], list(_nested_fields(rec, "arrays")))
        lines.append("}")
        lines.append("")

    lines.append(
        f"message {section.get('proto_message', 'RoutingProblem')} {{"
    )
    parent_scalars = list(_list(section, "scalars"))
    parent_arrays = list(_list(section, "arrays"))
    repeated = []
    for rec in _list(section, "member_records"):
        repeated.append(
            (
                rec["field_num"],
                f"  repeated {rec['message_type']} {rec['name']} = {rec['field_num']};",
            )
        )
    for rec in _list(section, "keyed_vectors"):
        repeated.append(
            (
                rec["field_num"],
                f"  repeated {rec['message_type']} {rec['name']} = {rec['field_num']};",
            )
        )
    for rec in _list(section, "keyed_records"):
        repeated.append(
            (
                rec["field_num"],
                f"  repeated {rec['message_type']} {rec['name']} = {rec['field_num']};",
            )
        )
    for rec in _optional_messages(section):
        repeated.append(
            (
                rec["field_num"],
                f"  {rec['message_type']} {rec['name']} = {rec['field_num']};",
            )
        )
    fields = [
        (f["field_num"], f"  {f['type']} {f['name']} = {f['field_num']};")
        for f in parent_scalars
    ]
    fields += [
        (f["field_num"], f"  {f['type']} {f['name']} = {f['field_num']};")
        for f in parent_arrays
    ]
    fields += repeated
    for _, line in sorted(fields, key=lambda item: item[0]):
        lines.append(line)
    lines.append("}")
    lines.append("")

    settings = registry.get("routing_settings") or {}
    lines.append(
        f"message {settings.get('proto_message', 'RoutingSolverSettings')} {{"
    )
    lines.append(
        "  // An omitted time_limit lets the solver derive its default."
    )
    lines.append("  // An explicit value, including 0, is honored.")
    for entry in settings.get("fields") or []:
        f = _parse(entry)
        lines.append(f"  {f['type']} {f['name']} = {f['field_num']};")
    lines.append("}")
    lines.append("")
    lines.append("message SolveVRPRequest {")
    lines.append("  RequestHeader header = 1;")
    lines.append(
        f"  {section.get('proto_message', 'RoutingProblem')} problem = 2;"
    )
    lines.append(
        f"  {settings.get('proto_message', 'RoutingSolverSettings')} settings = 3;"
    )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _cpp_copy_to_proto(field, src_expr, dst_expr):
    """Statements that append one C++ vector to a proto repeated field."""
    kind = field.get("proto_copy", "same")
    if kind == "u8_to_u32":
        return [
            f"  for (auto v : {src_expr}) {{",
            f"    {dst_expr}->Add(static_cast<uint32_t>(v));",
            "  }",
        ]
    if kind == "u8_to_bool":
        return [
            f"  for (auto v : {src_expr}) {{",
            f"    {dst_expr}->Add(v != 0);",
            "  }",
        ]
    return [f"  copy_vector_to_repeated({src_expr}, {dst_expr});"]


def _cpp_copy_from_proto(field, src_expr, dst_expr):
    kind = field.get("proto_copy", "same")
    if kind == "u8_to_u32":
        return [f"  copy_u32_to_u8({src_expr}, {dst_expr});"]
    if kind == "u8_to_bool":
        return [f"  copy_bool_to_u8({src_expr}, {dst_expr});"]
    return [f"  copy_repeated_to_vector({src_expr}, {dst_expr});"]


def _decode_fn(field):
    size = field.get("element_size")
    ftype = field.get("type", "repeated int32")
    if size == 1 or field.get("proto_copy") in ("u8_to_u32", "u8_to_bool"):
        return "decode_u8"
    if "float" in ftype:
        return "decode_f32"
    return "decode_i32"


def generate_routing_to_proto(registry):
    section = _section(registry)
    if not section:
        return ""
    lines = ["  pb->Clear();"]
    for f in _list(section, "scalars"):
        lines.append(f"  pb->set_{f['name']}(p.{f['member']});")
    lines.append("")

    for f in _list(section, "arrays"):
        lines.append("  if (include_other_arrays) {")
        for stmt in _cpp_copy_to_proto(
            f, f"p.{f['member']}", f"pb->mutable_{f['name']}()"
        ):
            lines.append("  " + stmt)
        lines.append("  }")
    lines.append("")

    message_bodies = {
        rec["message_type"]: rec for rec in _defined_records(section)
    }
    for rec in _list(section, "member_records"):
        body = message_bodies[rec["message_type"]]
        lines.append(f"  for (auto const& _entry : p.{rec['member']}) {{")
        lines.append(f"    auto* _out = pb->add_{rec['name']}();")
        for f in _nested_fields(body, "scalars"):
            cast = f.get("to_proto_cast")
            src = f"_entry.{f['member']}"
            if cast:
                lines.append(
                    f"    _out->set_{f['name']}(static_cast<{cast}>({src}));"
                )
            else:
                lines.append(f"    _out->set_{f['name']}({src});")
        array_guard = (
            "include_matrix_values"
            if any(
                f.get("matrix_values") for f in _nested_fields(body, "arrays")
            )
            else "include_other_arrays"
        )
        # A record can mix matrix arrays and ordinary arrays. Gate each array.
        for f in _nested_fields(body, "arrays"):
            guard = (
                "include_matrix_values"
                if f.get("matrix_values")
                else "include_other_arrays"
            )
            lines.append(f"    if ({guard}) {{")
            for stmt in _cpp_copy_to_proto(
                f, f"_entry.{f['member']}", f"_out->mutable_{f['name']}()"
            ):
                lines.append("    " + stmt)
            lines.append("    }")
        lines.append("  }")
        if array_guard:
            pass
    lines.append("")

    for rec in _list(section, "keyed_vectors"):
        key = rec["key"]
        values = rec["values"]
        lines.append(
            f"  for (auto const& [_key, _values] : p.{rec['member']}) {{"
        )
        lines.append(f"    auto* _out = pb->add_{rec['name']}();")
        lines.append(f"    _out->set_{key['name']}(_key);")
        lines.append("    if (include_other_arrays) {")
        for stmt in _cpp_copy_to_proto(
            values, "_values", f"_out->mutable_{values['name']}()"
        ):
            lines.append("    " + stmt)
        lines.append("    }")
        lines.append("  }")
    lines.append("")

    for rec in _list(section, "keyed_records"):
        key = rec["key"]
        lines.append(
            f"  for (auto const& [_key, _breaks] : p.{rec['member']}) {{"
        )
        lines.append(f"    auto* _out = pb->add_{rec['name']}();")
        lines.append(f"    _out->set_{key['name']}(_key);")
        lines.append("    for (auto const& _entry : _breaks) {")
        lines.append(f"      auto* _brk = _out->add_{rec['entry_field']}();")
        for f in _nested_fields(rec, "scalars"):
            lines.append(f"      _brk->set_{f['name']}(_entry.{f['member']});")
        for f in _nested_fields(rec, "arrays"):
            lines.append("      if (include_other_arrays) {")
            for stmt in _cpp_copy_to_proto(
                f, f"_entry.{f['member']}", f"_brk->mutable_{f['name']}()"
            ):
                lines.append("      " + stmt)
            lines.append("      }")
        lines.append("    }")
        lines.append("  }")
    lines.append("")

    for rec in _optional_messages(section):
        cond = rec.get("encode_when", "false")
        lines.append(f"  if ({cond}) {{")
        lines.append(f"    auto* _out = pb->mutable_{rec['name']}();")
        lines.append("    if (include_other_arrays) {")
        for f in _nested_fields(rec, "arrays"):
            for stmt in _cpp_copy_to_proto(
                f, f"p.{f['member']}", f"_out->mutable_{f['name']}()"
            ):
                lines.append("    " + stmt)
        lines.append("    }")
        lines.append("  }")
    lines.append("")
    return "\n".join(lines)


def generate_routing_from_proto(registry):
    section = _section(registry)
    if not section:
        return ""
    lines = [
        "  p = cuopt::routing::cpu_routing_problem_t{};",
        "  size_t _used_top = 0;",
        "  size_t _used_container = 0;",
        "  auto take_top = [&](int32_t id) -> const std::vector<uint8_t>* {",
        "    auto _it = arrays.find(id);",
        "    if (_it == arrays.end()) { return nullptr; }",
        "    ++_used_top;",
        "    return &_it->second;",
        "  };",
        "  auto take_container = [&](int32_t cfn, int32_t index, int32_t id) -> const std::vector<uint8_t>* {",
        "    cuopt::mathematical_optimization::container_array_key_t _key{cfn, index, id};",
        "    auto _it = container_arrays.find(_key);",
        "    if (_it == container_arrays.end()) { return nullptr; }",
        "    ++_used_container;",
        "    return &_it->second;",
        "  };",
        "  auto decode_i32 = [](const std::vector<uint8_t>& bytes, std::vector<int32_t>& dst) {",
        "    if (bytes.size() % sizeof(int32_t) != 0) {",
        '      throw std::invalid_argument("Chunked VRP int32 payload is not a multiple of sizeof(int32_t)");',
        "    }",
        "    dst.resize(bytes.size() / sizeof(int32_t));",
        "    if (!bytes.empty()) { std::memcpy(dst.data(), bytes.data(), bytes.size()); }",
        "  };",
        "  auto decode_f32 = [](const std::vector<uint8_t>& bytes, std::vector<float>& dst) {",
        "    if (bytes.size() % sizeof(float) != 0) {",
        '      throw std::invalid_argument("Chunked VRP float payload is not a multiple of sizeof(float)");',
        "    }",
        "    dst.resize(bytes.size() / sizeof(float));",
        "    if (!bytes.empty()) { std::memcpy(dst.data(), bytes.data(), bytes.size()); }",
        "  };",
        "  auto decode_u8 = [](const std::vector<uint8_t>& bytes, std::vector<uint8_t>& dst) {",
        "    dst = bytes;",
        "  };",
        "  auto apply_matrix = [&](const std::vector<uint8_t>& bytes, std::vector<float>& dst, const char* which, int index) {",
        "    if (bytes.size() % sizeof(float) != 0) {",
        '      throw std::invalid_argument("VRP matrix payload is not a multiple of sizeof(float)");',
        "    }",
        "    const int64_t got = static_cast<int64_t>(bytes.size() / sizeof(float));",
        "    const int64_t n = p.num_locations;",
        "    if (n <= 0) {",
        '      throw std::invalid_argument("Chunked VRP matrix requires num_locations > 0");',
        "    }",
        "    if (n > std::numeric_limits<int64_t>::max() / n) {",
        '      throw std::invalid_argument("VRP num_locations x num_locations overflows");',
        "    }",
        "    const int64_t expect = n * n;",
        "    if (got != expect) {",
        '      throw std::invalid_argument(std::string("Chunked VRP ") + which + " matrix " +',
        '                                  std::to_string(index) + " has " + std::to_string(got) +',
        '                                  " floats, expected " + std::to_string(expect));',
        "    }",
        "    dst.resize(static_cast<size_t>(got));",
        "    if (!bytes.empty()) { std::memcpy(dst.data(), bytes.data(), bytes.size()); }",
        "  };",
        "",
    ]
    for f in _list(section, "scalars"):
        lines.append(f"  p.{f['member']} = pb.{f['name']}();")
    lines.append("")

    from generate_conversions import _field_array_id_name

    for f in _list(section, "arrays"):
        aname = _field_array_id_name(f)
        decoder = _decode_fn(f)
        lines.append("  {")
        lines.append(
            f"    const auto* _chunk = take_top(static_cast<int32_t>(cuopt::remote::{aname}));"
        )
        lines.append("    if (_chunk) {")
        lines.append(f"      {decoder}(*_chunk, p.{f['member']});")
        lines.append("    } else {")
        for stmt in _cpp_copy_from_proto(
            f, f"pb.{f['name']}()", f"p.{f['member']}"
        ):
            lines.append("    " + stmt)
        lines.append("    }")
        lines.append("  }")
    lines.append("")

    message_bodies = {
        rec["message_type"]: rec for rec in _defined_records(section)
    }
    for rec in _list(section, "member_records"):
        body = message_bodies[rec["message_type"]]
        lines.append(
            f"  for (int _i = 0; _i < pb.{rec['name']}_size(); ++_i) {{"
        )
        lines.append(f"    auto const& _in = pb.{rec['name']}(_i);")
        lines.append(f"    {rec['cpp_type']} _out;")
        for f in _nested_fields(body, "scalars"):
            cast = f.get("from_proto_cast")
            src = f"_in.{f['name']}()"
            if cast:
                lines.append(
                    f"    _out.{f['member']} = static_cast<{cast}>({src});"
                )
            else:
                lines.append(f"    _out.{f['member']} = {src};")
        for f in _nested_fields(body, "arrays"):
            lines.append("    {")
            lines.append(
                f"      const auto* _chunk = take_container({rec['field_num']}, _i, {f['array_id']});"
            )
            lines.append("      if (_chunk) {")
            if f.get("matrix_values"):
                which = (
                    "cost"
                    if rec["name"] == "cost_matrices"
                    else "transit-time"
                )
                lines.append(
                    f'        apply_matrix(*_chunk, _out.{f["member"]}, "{which}", _i);'
                )
            else:
                lines.append(
                    f"        {_decode_fn(f)}(*_chunk, _out.{f['member']});"
                )
            lines.append("      } else {")
            for stmt in _cpp_copy_from_proto(
                f, f"_in.{f['name']}()", f"_out.{f['member']}"
            ):
                lines.append("    " + stmt)
            lines.append("      }")
            lines.append("    }")
        lines.append(f"    p.{rec['member']}.push_back(std::move(_out));")
        lines.append("  }")
    lines.append("")

    for rec in _list(section, "keyed_vectors"):
        key = rec["key"]
        values = rec["values"]
        lines.append(
            f"  for (int _i = 0; _i < pb.{rec['name']}_size(); ++_i) {{"
        )
        lines.append(f"    auto const& _in = pb.{rec['name']}(_i);")
        lines.append("    std::vector<int32_t> _values;")
        if "float" in values.get("type", ""):
            lines[-1] = "    std::vector<float> _values;"
        lines.append(
            f"    const auto* _chunk = take_container({rec['field_num']}, _i, {values['array_id']});"
        )
        lines.append("    if (_chunk) {")
        lines.append(f"      {_decode_fn(values)}(*_chunk, _values);")
        lines.append("    } else {")
        for stmt in _cpp_copy_from_proto(
            values, f"_in.{values['name']}()", "_values"
        ):
            lines.append("    " + stmt)
        lines.append("    }")
        lines.append(
            f"    p.{rec['member']}[_in.{key['name']}()] = std::move(_values);"
        )
        lines.append("  }")
    lines.append("")

    for rec in _list(section, "keyed_records"):
        key = rec["key"]
        lines.append("  {")
        lines.append("    int32_t _flat = 0;")
        lines.append(
            f"    for (int _i = 0; _i < pb.{rec['name']}_size(); ++_i) {{"
        )
        lines.append(f"      auto const& _group = pb.{rec['name']}(_i);")
        lines.append(f"      std::vector<{rec['entry_cpp_type']}> _breaks;")
        lines.append(
            f"      for (int _j = 0; _j < _group.{rec['entry_field']}_size(); ++_j, ++_flat) {{"
        )
        lines.append(
            f"        auto const& _in = _group.{rec['entry_field']}(_j);"
        )
        lines.append(f"        {rec['entry_cpp_type']} _out;")
        for f in _nested_fields(rec, "scalars"):
            lines.append(f"        _out.{f['member']} = _in.{f['name']}();")
        for f in _nested_fields(rec, "arrays"):
            lines.append(
                f"        const auto* _chunk = take_container({rec['field_num']}, _flat, {f['array_id']});"
            )
            lines.append("        if (_chunk) {")
            lines.append(
                f"          {_decode_fn(f)}(*_chunk, _out.{f['member']});"
            )
            lines.append("        } else {")
            for stmt in _cpp_copy_from_proto(
                f, f"_in.{f['name']}()", f"_out.{f['member']}"
            ):
                lines.append("        " + stmt)
            lines.append("        }")
        lines.append("        _breaks.push_back(std::move(_out));")
        lines.append("      }")
        lines.append(
            f"      p.{rec['member']}[_group.{key['name']}()] = std::move(_breaks);"
        )
        lines.append("    }")
        lines.append("  }")
    lines.append("")

    for rec in _optional_messages(section):
        arrays = list(_nested_fields(rec, "arrays"))
        lines.append("  {")
        chunk_vars = []
        for index, f in enumerate(arrays):
            var = f"_c{index}"
            chunk_vars.append((var, f))
            lines.append(
                f"    const auto* {var} = take_container({rec['field_num']}, 0, {f['array_id']});"
            )
        present = " || ".join(
            ["pb.has_" + rec["name"] + "()"] + [var for var, _ in chunk_vars]
        )
        lines.append(f"    if ({present}) {{")
        lines.append(
            f"      auto const* _msg = pb.has_{rec['name']}() ? &pb.{rec['name']}() : nullptr;"
        )
        for var, f in chunk_vars:
            lines.append(f"      if ({var}) {{")
            lines.append(f"        {_decode_fn(f)}(*{var}, p.{f['member']});")
            lines.append("      } else if (_msg) {")
            for stmt in _cpp_copy_from_proto(
                f, f"_msg->{f['name']}()", f"p.{f['member']}"
            ):
                lines.append("      " + stmt)
            lines.append("      }")
        lines.append("    }")
        lines.append("  }")
    lines.append("")
    lines.append(
        "  if (_used_top != arrays.size() || _used_container != container_arrays.size()) {"
    )
    lines.append(
        '    throw std::invalid_argument("Chunked VRP upload carries arrays that match no field");'
    )
    lines.append("  }")
    return "\n".join(lines)


def generate_routing_matrix_chunks(registry):
    section = _section(registry)
    if not section:
        return ""
    lines = []
    message_bodies = {
        rec["message_type"]: rec for rec in _defined_records(section)
    }
    for rec in _list(section, "member_records"):
        body = message_bodies[rec["message_type"]]
        for f in _nested_fields(body, "arrays"):
            if not f.get("matrix_values"):
                continue
            lines.append(
                f"  for (size_t _i = 0; _i < problem.{rec['member']}.size(); ++_i) {{"
            )
            lines.append(
                f"    chunk_container_typed_array(chunks, {rec['field_num']}, static_cast<int32_t>(_i), {f['array_id']},"
            )
            lines.append(
                f"                                 problem.{rec['member']}[_i].{f['member']}, upload_id, chunk_data_budget);"
            )
            lines.append("  }")
    return "\n".join(lines)


def generate_routing_nonmatrix_chunks(registry):
    from generate_conversions import _field_array_id_name

    section = _section(registry)
    if not section:
        return ""
    lines = []
    for f in _list(section, "arrays"):
        aname = _field_array_id_name(f)
        lines.append(
            f"  chunk_top_level_typed_array(chunks, static_cast<int32_t>(cuopt::remote::{aname}),"
        )
        lines.append(
            f"                              problem.{f['member']}, upload_id, chunk_data_budget);"
        )

    message_bodies = {
        rec["message_type"]: rec for rec in _defined_records(section)
    }
    for rec in _list(section, "member_records"):
        body = message_bodies[rec["message_type"]]
        for f in _nested_fields(body, "arrays"):
            if f.get("matrix_values"):
                continue
            lines.append(
                f"  for (size_t _i = 0; _i < problem.{rec['member']}.size(); ++_i) {{"
            )
            lines.append(
                f"    chunk_container_typed_array(chunks, {rec['field_num']}, static_cast<int32_t>(_i), {f['array_id']},"
            )
            lines.append(
                f"                                 problem.{rec['member']}[_i].{f['member']}, upload_id, chunk_data_budget);"
            )
            lines.append("  }")

    for rec in _list(section, "keyed_vectors"):
        values = rec["values"]
        lines.append("  {")
        lines.append("    int32_t _i = 0;")
        lines.append(
            f"    for (auto const& [_key, _values] : problem.{rec['member']}) {{"
        )
        lines.append(
            f"      chunk_container_typed_array(chunks, {rec['field_num']}, _i, {values['array_id']}, _values, upload_id, chunk_data_budget);"
        )
        lines.append("      ++_i;")
        lines.append("    }")
        lines.append("  }")

    for rec in _list(section, "keyed_records"):
        lines.append("  {")
        lines.append("    int32_t _flat = 0;")
        lines.append(
            f"    for (auto const& [_key, _breaks] : problem.{rec['member']}) {{"
        )
        lines.append("      for (auto const& _entry : _breaks) {")
        for f in _nested_fields(rec, "arrays"):
            lines.append(
                f"        chunk_container_typed_array(chunks, {rec['field_num']}, _flat, {f['array_id']}, _entry.{f['member']}, upload_id, chunk_data_budget);"
            )
        lines.append("        ++_flat;")
        lines.append("      }")
        lines.append("    }")
        lines.append("  }")

    for rec in _optional_messages(section):
        for f in _nested_fields(rec, "arrays"):
            lines.append(
                f"  chunk_container_typed_array(chunks, {rec['field_num']}, 0, {f['array_id']}, problem.{f['member']}, upload_id, chunk_data_budget);"
            )
    return "\n".join(lines)


def emit_routing(registry, outdir, header, write_file):
    import os

    if not _section(registry):
        return
    write_file(
        os.path.join(outdir, "cuopt_routing.proto"),
        generate_routing_proto(registry),
    )
    write_file(
        os.path.join(outdir, "generated_routing_problem_to_proto.inc"),
        header + generate_routing_to_proto(registry) + "\n",
    )
    write_file(
        os.path.join(outdir, "generated_routing_problem_from_proto.inc"),
        header + generate_routing_from_proto(registry) + "\n",
    )
    write_file(
        os.path.join(outdir, "generated_routing_matrix_chunks.inc"),
        header + generate_routing_matrix_chunks(registry) + "\n",
    )
    write_file(
        os.path.join(outdir, "generated_routing_nonmatrix_chunks.inc"),
        header + generate_routing_nonmatrix_chunks(registry) + "\n",
    )
