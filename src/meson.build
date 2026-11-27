inc = include_directories('.')

libcistern = static_library('cistern',
  files('align.cc', 'direct_file.cc', 'env.cc', 'layout.cc', 'restore.cc', 'ring.cc', 'state.cc',
        'tensor_table.cc', 'writer.cc'),
  include_directories: inc,
  dependencies: [liburing, json],
  gnu_symbol_visibility: 'hidden')

cistern_dep = declare_dependency(
  link_with: libcistern,
  include_directories: inc,
  dependencies: [liburing, json])
