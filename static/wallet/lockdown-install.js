(() => {
  if (typeof globalThis.lockdown !== "function") {
    globalThis.lockdown = function lockdown() {
      return undefined;
    };
  }

  if (typeof globalThis.harden !== "function") {
    globalThis.harden = function harden(value) {
      return value;
    };
  }
})();

//# sourceMappingURL=lockdown-install.js.map