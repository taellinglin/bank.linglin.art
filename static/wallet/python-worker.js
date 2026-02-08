self.pyodideUrl = null;
self.appPackageUrl = null;
self.micropipIncludePre = false;
self.pythonModuleName = null;
self.documentUrl = null;
self.initialized = false;
self.flet_js = {}; // namespace for Python global functions
self.flet_js.report_progress = (message, progress = null) => {
    self.postMessage({
        type: "progress",
        message,
        progress
    });
};

self.initPyodide = async function () {
    try {
        // Provide browser-like globals for Pyodide running in a worker
        if (typeof self.window === "undefined") {
            self.window = self;
        }
        if (typeof self.localStorage === "undefined") {
            const _storage = new Map();
            self.localStorage = {
                getItem: (key) => (_storage.has(key) ? _storage.get(key) : null),
                setItem: (key, value) => { _storage.set(key, String(value)); },
                removeItem: (key) => { _storage.delete(key); },
                clear: () => { _storage.clear(); }
            };
        }
        if (typeof self.sessionStorage === "undefined") {
            const _session = new Map();
            self.sessionStorage = {
                getItem: (key) => (_session.has(key) ? _session.get(key) : null),
                setItem: (key, value) => { _session.set(key, String(value)); },
                removeItem: (key) => { _session.delete(key); },
                clear: () => { _session.clear(); }
            };
        }
        self.flet_js.report_progress("Loading Python runtime...");
        importScripts(self.pyodideUrl);
        self.pyodide = await loadPyodide();
        self.pyodide.setStdout({
            batched: (text) => {
                try {
                    self.postMessage({ type: "pylog", level: "stdout", message: text });
                } catch (_) {}
            }
        });
        self.pyodide.setStderr({
            batched: (text) => {
                try {
                    self.postMessage({ type: "pylog", level: "stderr", message: text });
                } catch (_) {}
            }
        });
        self.flet_js.report_progress("Initializing Pyodide...");
        self.pyodide.registerJsModule("flet_js", flet_js);
        self.pyodide.globals.set("app_package_url", self.appPackageUrl);
        self.pyodide.globals.set("python_module_name", self.pythonModuleName);
        self.pyodide.globals.set("micropip_include_pre", self.micropipIncludePre);
        flet_js.documentUrl = documentUrl;
        self.flet_js.report_progress("Loading micropip...");
        await self.pyodide.loadPackage("micropip");

        // Load unvendored stdlib modules commonly required by dependencies
        const stdlibPackages = ["ssl", "sqlite3"];
        for (const pkg of stdlibPackages) {
            self.flet_js.report_progress(`Loading ${pkg} module...`);
            try {
                await self.pyodide.loadPackage(pkg);
            } catch (e) {
                console.warn(`Failed to load stdlib package: ${pkg}`, e);
            }
        }
        self.flet_js.report_progress("Preparing app...");
        await self.pyodide.runPythonAsync(`
        import flet_js, micropip, os, runpy, sys, traceback, warnings
        from pyodide.http import pyfetch

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        py_args = flet_js.args.to_py() if flet_js.args else None

        if "app_package_url" in py_args:
            app_package_url = py_args["app_package_url"]

        if app_package_url is None:
            app_package_url = "assets/app/app.zip"

        if "python_module_name" in py_args:
            python_module_name = py_args["python_module_name"]

        if python_module_name is None:
            python_module_name = "main"

        if "micropip_include_pre" in py_args:
            micropip_include_pre = py_args["micropip_include_pre"]

        if micropip_include_pre is None:
            micropip_include_pre = False

        print("python_module_name:", python_module_name)
        print("micropip_include_pre:", micropip_include_pre)

        if "script" not in py_args:
            print("Downloading app archive")
            flet_js.report_progress("Downloading app archive...")
            response = await pyfetch(app_package_url)
            await response.unpack_archive()
        else:
            print("Saving script to a file")
            flet_js.report_progress("Saving script...")
            with open(f"{python_module_name}.py", "w") as f:
                f.write(py_args["script"]);

        pkgs_path = "__pypackages__"
        if os.path.exists(pkgs_path):
            print(f"Adding {pkgs_path} to sys.path")
            sys.path.insert(0, pkgs_path)

        if os.path.exists("requirements.txt"):
            with open("requirements.txt", "r") as f:
                raw_deps = [line.rstrip() for line in f]
                deps = []
                for dep in raw_deps:
                    if not dep or dep.lstrip().startswith("#"):
                        continue
                    # skip lunalib here; we will enforce the version explicitly below
                    if dep.strip().lower().startswith("lunalib"):
                        continue
                    # skip packaging pins; pyodide ships packaging already
                    if dep.strip().lower().startswith("packaging"):
                        continue
                    deps.append(dep)
                print("Loading requirements.txt:", raw_deps)
                flet_js.report_progress("Installing requirements.txt packages...")
                if deps:
                    await micropip.install(deps, pre=micropip_include_pre)

        if "dependencies" in py_args:
            flet_js.report_progress("Installing dependencies...")
            await micropip.install(py_args["dependencies"], pre=micropip_include_pre)

        # Ensure lunalib version (Pyodide cannot use platform wheels)
        flet_js.report_progress("Ensuring lunalib==2.6.6...")
        try:
            import importlib.metadata as _metadata
            _installed_ver = _metadata.version("lunalib")
        except Exception:
            _installed_ver = None

        if _installed_ver != "2.6.6":
            # Remove existing lunalib files if present to avoid version conflict
            import glob, shutil
            for p in list(sys.path):
                if not p:
                    continue
                for pattern in (
                    "lunalib*",
                    "luna_lib*",
                    "lunalib-*.dist-info",
                    "lunalib-*.egg-info",
                ):
                    for target in glob.glob(os.path.join(p, pattern)):
                        try:
                            if os.path.isdir(target):
                                shutil.rmtree(target)
                            else:
                                os.remove(target)
                        except Exception:
                            pass

            # Fetch sdist from PyPI and add to sys.path (pure Python fallback)
            import json, tarfile, io
            flet_js.report_progress("Fetching lunalib sdist...")
            pypi_resp = await pyfetch("https://pypi.org/pypi/lunalib/2.6.6/json")
            pypi_data = await pypi_resp.json()
            sdist_url = None
            for u in pypi_data.get("urls", []):
                if u.get("packagetype") == "sdist":
                    sdist_url = u.get("url")
                    break
            if not sdist_url:
                raise RuntimeError("lunalib sdist not found on PyPI")
            sdist_resp = await pyfetch(sdist_url)
            sdist_bytes = await sdist_resp.bytes()
            extract_root = "vendor/lunalib-2.6.6"
            os.makedirs(extract_root, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(sdist_bytes), mode="r:gz") as tf:
                tf.extractall(extract_root)
            # add extracted top-level folder to sys.path
            subdirs = [d for d in os.listdir(extract_root) if os.path.isdir(os.path.join(extract_root, d))]
            if subdirs:
                sys.path.insert(0, os.path.join(extract_root, subdirs[0]))
            else:
                sys.path.insert(0, extract_root)

        # Execute app
        flet_js.report_progress("Starting app...")

        # Patch lunalib MempoolManager to avoid missing attribute in Pyodide
        try:
            import lunalib.core.mempool as _mempool
            if not hasattr(_mempool.MempoolManager, "verbose"):
                setattr(_mempool.MempoolManager, "verbose", False)
        except Exception:
            pass

        # Disable threading in Pyodide (no native threads)
        import threading
        def _pyodide_no_thread_start(self):
            return
        threading.Thread.start = _pyodide_no_thread_start

        # Avoid un-awaited coroutine warnings for page.window.center()
        try:
            import asyncio
            import flet as ft
            from flet.core.window import Window
            _orig_center = Window.center
            def _center_noawait(self, *args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                    return loop.create_task(_orig_center(self, *args, **kwargs))
                except Exception:
                    return None
            Window.center = _center_noawait
        except Exception:
            pass

        try:
            runpy.run_module(python_module_name, run_name="__main__")
        except Exception:
            traceback.print_exc()
            raise
      `);
        await self.flet_js.start_connection(self.receiveCallback);
        self.postMessage("initialized");
    } catch (error) {
        const details = [
            error?.toString?.() || "Unknown error",
            error?.message || "",
            error?.stack || ""
        ].filter(Boolean).join("\n");
        console.error("Python worker init error:", details);
        self.postMessage(details);
    }
};

self.receiveCallback = (message) => {
    self.postMessage(message.toJs());
}

self.onmessage = async (event) => {
    // run only once
    if (!self.initialized) {
        self.initialized = true;
        self.pyodideUrl = event.data.pyodideUrl;
        self.flet_js.args = event.data.args;
        self.documentUrl = event.data.documentUrl;
        self.appPackageUrl = event.data.appPackageUrl;
        self.micropipIncludePre = event.data.micropipIncludePre;
        self.pythonModuleName = event.data.pythonModuleName;
        await self.initPyodide();
    } else {
        // message
        if (typeof flet_js.send === "function") {
            flet_js.send(event.data);
        }
    }
};
