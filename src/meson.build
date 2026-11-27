inc = include_directories('.')

libpitstop = static_library('pitstop',
  files('align.cc', 'direct_file.cc', 'env.cc', 'layout.cc', 'restore.cc', 'ring.cc', 'state.cc',
        'tensor_table.cc', 'writer.cc'),
  include_directories: inc,
  dependencies: [liburing, json],
  gnu_symbol_visibility: 'hidden')

pitstop_dep = declare_dependency(
  link_with: libpitstop,
  include_directories: inc,
  dependencies: [liburing, json])
