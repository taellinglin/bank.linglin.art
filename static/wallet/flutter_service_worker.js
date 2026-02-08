'use strict';
const MANIFEST = 'flutter-app-manifest';
const TEMP = 'flutter-temp-cache';
const CACHE_NAME = 'flutter-app-cache';

const RESOURCES = {"assets/app/app.zip": "96b7f02f5202ae19b62762ab5e227842",
"assets/app/app.zip.hash": "acbf45b4cc1627c5c05e5e4199c0bf9e",
"assets/AssetManifest.bin": "5e4c69c57b629fa3c0fbf78c71c4db94",
"assets/AssetManifest.bin.json": "992ba04d5892193176350dad1f95a2ec",
"assets/FontManifest.json": "6d7513ce1c88ccff09eb0c72d3685bdc",
"assets/fonts/MaterialIcons-Regular.otf": "f34ace52ea74c95e26949fab4870ac22",
"assets/fonts/roboto.woff2": "e507bd45228483ae2f864d36f26bb43e",
"assets/NOTICES": "ba69bbc655ac60decae302bf7cdf03fe",
"assets/packages/cupertino_icons/assets/CupertinoIcons.ttf": "6323a28c4d27ae6070923bcb643dc985",
"assets/packages/wakelock_plus/assets/no_sleep.js": "7748a45cd593f33280669b29c2c8919a",
"assets/shaders/ink_sparkle.frag": "ecc85a2e95f5e9f53123dcaf8cb9b6ce",
"assets/shaders/stretch_effect.frag": "40d68efbbf360632f614c731219e95f0",
"canvaskit/canvaskit.js": "86e461cf471c1640fd2b461ece4589df",
"canvaskit/canvaskit.js.symbols": "68eb703b9a609baef8ee0e413b442f33",
"canvaskit/canvaskit.wasm": "efeeba7dcc952dae57870d4df3111fad",
"canvaskit/chromium/canvaskit.js": "34beda9f39eb7d992d46125ca868dc61",
"canvaskit/chromium/canvaskit.js.symbols": "5a23598a2a8efd18ec3b60de5d28af8f",
"canvaskit/chromium/canvaskit.wasm": "64a386c87532ae52ae041d18a32a3635",
"canvaskit/skwasm.js": "f2ad9363618c5f62e813740099a80e63",
"canvaskit/skwasm.js.symbols": "80806576fa1056b43dd6d0b445b4b6f7",
"canvaskit/skwasm.wasm": "f0dfd99007f989368db17c9abeed5a49",
"canvaskit/skwasm_heavy.js": "740d43a6b8240ef9e23eed8c48840da4",
"canvaskit/skwasm_heavy.js.symbols": "0755b4fb399918388d71b59ad390b055",
"canvaskit/skwasm_heavy.wasm": "b0be7910760d205ea4e011458df6ee01",
"canvaskit/skwasm_st.js": "d1326ceef381ad382ab492ba5d96f04d",
"canvaskit/skwasm_st.js.symbols": "c7e7aac7cd8b612defd62b43e3050bdd",
"canvaskit/skwasm_st.wasm": "56c3973560dfcbf28ce47cebe40f3206",
"favicon.png": "fd41265073bab10e9a753d45bd49db75",
"flutter.js": "24bc71911b75b5f8135c949e27a2984e",
"flutter.js.map": "493b39420f09daa62e485b78a7ff50ba",
"flutter_bootstrap.js": "5d9182611fabd807792eb2b23c2ddf8e",
"icons/apple-touch-icon-192.png": "8cf0d5162941f467a77f023c414a1812",
"icons/Icon-192.png": "720f9a4c3873ce75e7d775303ea32565",
"icons/Icon-512.png": "d644dd264d323c25ecf65d73d9d5b75d",
"icons/Icon-maskable-192.png": "720f9a4c3873ce75e7d775303ea32565",
"icons/Icon-maskable-512.png": "d644dd264d323c25ecf65d73d9d5b75d",
"icons/loading-animation.png": "41a96047dbd2463a50c46ad3bf6ff158",
"index.html": "d005df8666b9a7ea835c23cf83c90d00",
"lockdown-install.js": "46c9404759da92f167563c142c4535cc",
"/": "d005df8666b9a7ea835c23cf83c90d00",
"main.dart.js": "e50b1264a767bf2176c3e0a1dc2195a9",
"main.dart.mjs": "9a2bf71b6ffa0566d57df0ffaa046a62",
"main.dart.wasm": "947a5e2323875ebc59e2c8bd764865bb",
"manifest.json": "6369f5a0703bd990a54f41d01e91c5c9",
"pyodide/ffi.d.ts": "e40213f539be775d0924e4aa348ec4f7",
"pyodide/micropip-0.8.0-py3-none-any.whl": "b132a43045c127404f00f781d32f3048",
"pyodide/package.json": "e7dad597b3686bf79bb01240086a4de8",
"pyodide/packaging-24.2-py3-none-any.whl": "ba8472e04cb67139842aa03ff5921358",
"pyodide/pyodide-lock.json": "c514c0f3480fe7388346a9106cc56d95",
"pyodide/pyodide.asm.js": "31daa2b26f2436587ab55425451df592",
"pyodide/pyodide.asm.wasm": "ba116948a682d77867d1e34d9e837614",
"pyodide/pyodide.d.ts": "13cfd754c98bc09d35b15f30661623c8",
"pyodide/pyodide.js": "3f5a03308cbaf16edcf3a456673ea441",
"pyodide/pyodide.mjs": "d3c7620427e7f434afc90983bb2219b6",
"pyodide/python_stdlib.zip": "ba7bdcbf412690e702e7f1e0997382ed",
"python-worker.js": "26eb131f3acb5ce232fea72da957e8ce",
"python.js": "352c5261eadd3cc73ac082984266c0fc",
"splash/img/dark-1x.png": "ebab7302660ca6701be00843be476caf",
"splash/img/dark-2x.png": "d644dd264d323c25ecf65d73d9d5b75d",
"splash/img/dark-3x.png": "40c26f198255b3fdfbb211419643b2ac",
"splash/img/dark-4x.png": "0c45fcdca0a01fd0e583949f7776326b",
"splash/img/light-1x.png": "ebab7302660ca6701be00843be476caf",
"splash/img/light-2x.png": "d644dd264d323c25ecf65d73d9d5b75d",
"splash/img/light-3x.png": "40c26f198255b3fdfbb211419643b2ac",
"splash/img/light-4x.png": "0c45fcdca0a01fd0e583949f7776326b",
"version.json": "abfc9ce34968c404f7b238c5f64c62f2"};
// The application shell files that are downloaded before a service worker can
// start.
const CORE = ["main.dart.js",
"main.dart.wasm",
"main.dart.mjs",
"index.html",
"flutter_bootstrap.js",
"lockdown-install.js",
"assets/AssetManifest.bin.json",
"assets/FontManifest.json"];

// During install, the TEMP cache is populated with the application shell files.
self.addEventListener("install", (event) => {
  self.skipWaiting();
  return event.waitUntil(
    caches.open(TEMP).then((cache) => {
      return cache.addAll(
        CORE.map((value) => new Request(value, {'cache': 'reload'})));
    })
  );
});
// During activate, the cache is populated with the temp files downloaded in
// install. If this service worker is upgrading from one with a saved
// MANIFEST, then use this to retain unchanged resource files.
self.addEventListener("activate", function(event) {
  return event.waitUntil(async function() {
    try {
      var contentCache = await caches.open(CACHE_NAME);
      var tempCache = await caches.open(TEMP);
      var manifestCache = await caches.open(MANIFEST);
      var manifest = await manifestCache.match('manifest');
      // When there is no prior manifest, clear the entire cache.
      if (!manifest) {
        await caches.delete(CACHE_NAME);
        contentCache = await caches.open(CACHE_NAME);
        for (var request of await tempCache.keys()) {
          var response = await tempCache.match(request);
          await contentCache.put(request, response);
        }
        await caches.delete(TEMP);
        // Save the manifest to make future upgrades efficient.
        await manifestCache.put('manifest', new Response(JSON.stringify(RESOURCES)));
        // Claim client to enable caching on first launch
        self.clients.claim();
        return;
      }
      var oldManifest = await manifest.json();
      var origin = self.location.origin;
      for (var request of await contentCache.keys()) {
        var key = request.url.substring(origin.length + 1);
        if (key == "") {
          key = "/";
        }
        // If a resource from the old manifest is not in the new cache, or if
        // the MD5 sum has changed, delete it. Otherwise the resource is left
        // in the cache and can be reused by the new service worker.
        if (!RESOURCES[key] || RESOURCES[key] != oldManifest[key]) {
          await contentCache.delete(request);
        }
      }
      // Populate the cache with the app shell TEMP files, potentially overwriting
      // cache files preserved above.
      for (var request of await tempCache.keys()) {
        var response = await tempCache.match(request);
        await contentCache.put(request, response);
      }
      await caches.delete(TEMP);
      // Save the manifest to make future upgrades efficient.
      await manifestCache.put('manifest', new Response(JSON.stringify(RESOURCES)));
      // Claim client to enable caching on first launch
      self.clients.claim();
      return;
    } catch (err) {
      // On an unhandled exception the state of the cache cannot be guaranteed.
      console.error('Failed to upgrade service worker: ' + err);
      await caches.delete(CACHE_NAME);
      await caches.delete(TEMP);
      await caches.delete(MANIFEST);
    }
  }());
});
// The fetch handler redirects requests for RESOURCE files to the service
// worker cache.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== 'GET') {
    return;
  }
  var origin = self.location.origin;
  var key = event.request.url.substring(origin.length + 1);
  // Redirect URLs to the index.html
  if (key.indexOf('?v=') != -1) {
    key = key.split('?v=')[0];
  }
  if (key === 'lockdown-install.js' || key.endsWith('/lockdown-install.js')) {
    event.respondWith(new Response(
      "(() => {\n" +
      "  if (typeof globalThis.lockdown !== \\\"function\\\") {\n" +
      "    globalThis.lockdown = function lockdown() {\n" +
      "      return undefined;\n" +
      "    };\n" +
      "  }\n\n" +
      "  if (typeof globalThis.harden !== \\\"function\\\") {\n" +
      "    globalThis.harden = function harden(value) {\n" +
      "      return value;\n" +
      "    };\n" +
      "  }\n" +
      "})();\n",
      { headers: { 'Content-Type': 'text/javascript' } }
    ));
    return;
  }
  if (event.request.url == origin || event.request.url.startsWith(origin + '/#') || key == '') {
    key = '/';
  }
  // If the URL is not the RESOURCE list then return to signal that the
  // browser should take over.
  if (!RESOURCES[key]) {
    return;
  }
  // If the URL is the index.html, perform an online-first request.
  if (key == '/') {
    return onlineFirst(event);
  }
  event.respondWith(caches.open(CACHE_NAME)
    .then((cache) =>  {
      return cache.match(event.request).then((response) => {
        // Either respond with the cached resource, or perform a fetch and
        // lazily populate the cache only if the resource was successfully fetched.
        return response || fetch(event.request).then((response) => {
          if (response && Boolean(response.ok)) {
            cache.put(event.request, response.clone());
          }
          return response;
        });
      })
    })
  );
});
self.addEventListener('message', (event) => {
  // SkipWaiting can be used to immediately activate a waiting service worker.
  // This will also require a page refresh triggered by the main worker.
  if (event.data === 'skipWaiting') {
    self.skipWaiting();
    return;
  }
  if (event.data === 'downloadOffline') {
    downloadOffline();
    return;
  }
});
// Download offline will check the RESOURCES for all files not in the cache
// and populate them.
async function downloadOffline() {
  var resources = [];
  var contentCache = await caches.open(CACHE_NAME);
  var currentContent = {};
  for (var request of await contentCache.keys()) {
    var key = request.url.substring(origin.length + 1);
    if (key == "") {
      key = "/";
    }
    currentContent[key] = true;
  }
  for (var resourceKey of Object.keys(RESOURCES)) {
    if (!currentContent[resourceKey]) {
      resources.push(resourceKey);
    }
  }
  return contentCache.addAll(resources);
}
// Attempt to download the resource online before falling back to
// the offline cache.
function onlineFirst(event) {
  return event.respondWith(
    fetch(event.request).then((response) => {
      return caches.open(CACHE_NAME).then((cache) => {
        cache.put(event.request, response.clone());
        return response;
      });
    }).catch((error) => {
      return caches.open(CACHE_NAME).then((cache) => {
        return cache.match(event.request).then((response) => {
          if (response != null) {
            return response;
          }
          throw error;
        });
      });
    })
  );
}
