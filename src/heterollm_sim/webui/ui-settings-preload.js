// Apply local-only appearance before the stylesheet is parsed to avoid a theme flash.
(() => {
  const defaults = {
    language: "zh-CN",
    fontScale: 120,
    theme: "graphite",
    workspaceBackground: null,
    compact: false,
    topologyGrid: true,
    reduceMotion: false,
  };
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem("heterollm-lab:ui-settings:v3") || "{}");
  } catch (_error) {
    saved = {};
  }
  const savedSettings = saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
  const settings = { ...defaults, ...savedSettings };
  const root = document.documentElement;
  const language = ["zh-CN", "en"].includes(settings.language) ? settings.language : defaults.language;
  root.lang = language;
  root.dataset.language = language;
  const requestedFontScale = Number(savedSettings.fontScale ?? defaults.fontScale);
  const fontScale = Number.isFinite(requestedFontScale)
    ? Math.min(200, Math.max(80, Math.round(requestedFontScale)))
    : defaults.fontScale;
  const fontRatio = fontScale / 100;
  const layoutScale = Math.min(4, Math.max(0.85, 1 + (fontRatio - 1) * 0.75));
  root.dataset.fontScale = String(fontScale);
  root.dataset.fontBand = fontScale >= 190 ? "extreme" : fontScale > 150 ? "large" : "normal";
  root.style.setProperty("--font-scale", fontRatio.toFixed(2));
  root.style.setProperty("--layout-scale", layoutScale.toFixed(3));
  const themes = ["graphite", "bluegray", "black", "ivory", "mist", "softgray"];
  root.dataset.theme = themes.includes(settings.theme)
    ? settings.theme
    : defaults.theme;
  root.style.colorScheme = ["ivory", "mist", "softgray"].includes(root.dataset.theme) ? "light" : "dark";
  root.dataset.density = settings.compact === true ? "compact" : "standard";
  root.dataset.topologyGrid = settings.topologyGrid === false ? "off" : "on";
  root.dataset.reduceMotion = settings.reduceMotion === true ? "true" : "false";
  if (/^#[0-9a-f]{6}$/i.test(settings.workspaceBackground || "")) {
    root.dataset.customWorkspace = "true";
    root.style.setProperty("--workspace-bg-custom", settings.workspaceBackground);
  }
})();
