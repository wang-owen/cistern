# The two baselines (buffered write()+fsync, pwrite+O_DIRECT) and a one-shot CLI around them.
baselines_inc = include_directories('..')  # so "baselines/baselines.h" resolves

libbaselines = static_library('cistern_baselines', files('baselines.cc'),
  include_directories: baselines_inc,
  dependencies: [cistern_dep])

baselines_dep = declare_dependency(
  link_with: libbaselines,
  include_directories: baselines_inc,
  dependencies: [cistern_dep])

executable('cistern-baseline', files('baseline_cli.cc'),
  dependencies: [baselines_dep])
