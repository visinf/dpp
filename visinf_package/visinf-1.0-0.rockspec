package = "visinf"
version = "1.0-0"

source = {
   url = "-",
   tag = "master"
}

description = {
   summary = "VISINF torch nn repository",
   detailed = [[
TU-Darmstadt, Visual Inference group libraries
   ]],
   homepage = "--",
   license = "BSD"
}

dependencies = {
   "torch >= 8.0",
   "nn",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
