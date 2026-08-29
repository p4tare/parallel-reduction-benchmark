# v2.6.2 hotfix

## Fixed: YAML `on`/`off` parsed as booleans

PyYAML's default `SafeLoader` uses YAML 1.1 implicit boolean rules. As a result,

```yaml
build:
  enable_cuda: on
```

was parsed as Python `True`, while the configuration schema correctly expects one of the strings `auto`, `on`, or `off`.

v2.6.2 introduces a project-local safe loader with YAML 1.2-style boolean semantics: only `true` and `false` are implicitly parsed as booleans. Therefore unquoted `on` and `off` remain strings and work with `build.enable_cuda`.

This fixes the root cause globally; quoting `"on"`/`"off"` remains valid and is still recommended in hand-written configs for maximum portability.

Two regression tests cover `enable_cuda: on`, `enable_cuda: off`, and ordinary `false` boolean fields.
