"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const styles = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function blockFromOpeningBrace(source, openIndex) {
  assert.equal(source[openIndex], "{", "block must start at an opening brace");
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(openIndex + 1, index);
    }
  }
  assert.fail("unterminated block");
}

function functionSource(name) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} must exist`);
  const open = app.indexOf("{", start);
  assert.ok(open > start, `${name} must have a body`);
  return app.slice(start, open + 1) + blockFromOpeningBrace(app, open) + "}";
}

function sourceBetweenFunctions(startName, endName) {
  const start = app.indexOf(`function ${startName}(`);
  const end = app.indexOf(`\nfunction ${endName}(`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

function normalizeCss(value) {
  return String(value).replace(/\s+/gu, " ").trim();
}

function mediaBlocksMatching(predicate) {
  const blocks = [];
  for (const match of styles.matchAll(/@media\s+([^{]+)\{/gu)) {
    const condition = normalizeCss(match[1]);
    if (!predicate(condition)) continue;
    const open = styles.indexOf("{", match.index);
    blocks.push({ condition, block: blockFromOpeningBrace(styles, open) });
  }
  return blocks;
}

function mediaBlockText(predicate, context) {
  const blocks = mediaBlocksMatching(predicate);
  assert.ok(blocks.length, `missing ${context} media block`);
  return blocks.map((item) => item.block).join("\n");
}

function conditionHasMaxWidth(condition, px) {
  const escaped = escapeRegExp(String(px));
  return new RegExp(`max-width\\s*:\\s*${escaped}px`, "u").test(condition)
    || new RegExp(`width\\s*<=\\s*${escaped}px`, "u").test(condition)
    || new RegExp(`${escaped}px\\s*>=\\s*width`, "u").test(condition);
}

function conditionHasMinWidth(condition, px) {
  const escaped = escapeRegExp(String(px));
  return new RegExp(`min-width\\s*:\\s*${escaped}px`, "u").test(condition)
    || new RegExp(`width\\s*>=\\s*${escaped}px`, "u").test(condition)
    || new RegExp(`${escaped}px\\s*<=\\s*width`, "u").test(condition);
}

function cssRules(source) {
  return Array.from(source.matchAll(/([^{}]+)\{([^{}]*)\}/gu), (match) => ({
    selector: normalizeCss(match[1]),
    body: match[2],
  })).filter((rule) => !rule.selector.startsWith("@"));
}

function selectorHasClass(selector, className) {
  return new RegExp(`\\.${escapeRegExp(className)}(?![-_a-zA-Z0-9])`, "u").test(selector);
}

function declarationValue(body, property) {
  const match = body.match(new RegExp(`${escapeRegExp(property)}\\s*:\\s*([^;]+);`, "u"));
  return match ? normalizeCss(match[1]) : null;
}

function declarationValuesForClasses(source, classNames, property) {
  return cssRules(source)
    .filter((rule) => classNames.every((className) => selectorHasClass(rule.selector, className)))
    .map((rule) => declarationValue(rule.body, property))
    .filter(Boolean);
}

function lastDeclarationForClasses(source, classNames, property, context) {
  const values = declarationValuesForClasses(source, classNames, property);
  assert.ok(values.length, `${context}: missing ${property} for .${classNames.join(".")}`);
  return values.at(-1);
}

function assertDeclaration(source, classNames, property, pattern, context) {
  const value = lastDeclarationForClasses(source, classNames, property, context);
  assert.match(value, pattern, `${context}: expected ${property} to match ${pattern}, got ${value}`);
}

function assertDeclarationInCascade(source, fallbackSource, classNames, property, pattern, context) {
  const localValues = declarationValuesForClasses(source, classNames, property);
  const cascadeValues = localValues.length ? localValues : declarationValuesForClasses(fallbackSource, classNames, property);
  assert.ok(cascadeValues.length, `${context}: missing ${property} for .${classNames.join(".")}`);
  const value = cascadeValues.at(-1);
  assert.match(value, pattern, `${context}: expected ${property} to match ${pattern}, got ${value}`);
}

function assertSomeDeclaration(source, classNames, properties, pattern, context) {
  const values = properties.flatMap((property) => (
    declarationValuesForClasses(source, classNames, property).map((value) => `${property}: ${value}`)
  ));
  assert.ok(values.length, `${context}: missing ${properties.join(" or ")} for .${classNames.join(".")}`);
  assert.ok(
    values.some((entry) => pattern.test(entry)),
    `${context}: expected one of ${values.join("; ")} to match ${pattern}`,
  );
}

function splitTopLevelWhitespace(value) {
  const tokens = [];
  let token = "";
  let depth = 0;
  for (const char of value.trim()) {
    if (char === "(" || char === "[") depth += 1;
    else if (char === ")" || char === "]") depth -= 1;
    if (/\s/u.test(char) && depth === 0) {
      if (token) tokens.push(token);
      token = "";
    } else {
      token += char;
    }
  }
  if (token) tokens.push(token);
  return tokens;
}

function gridTrackCount(value) {
  const normalized = normalizeCss(value);
  const repeat = normalized.match(/^repeat\(\s*(\d+)\s*,/u);
  if (repeat) return Number(repeat[1]);
  return splitTopLevelWhitespace(normalized).length;
}

function openingHeadingTagsWithConceptHelp() {
  return Array.from(
    (html + "\n" + app).matchAll(/<h([1-6])\b(?=[^>]*\bdata-concept-help\s*=)[^>]*>/gu),
    (match) => match[0],
  );
}

function makeFakeElement(tagName, {
  text = "",
  dataset = {},
  className = "",
  parent = null,
  allElements,
}) {
  const classNames = new Set();
  const attributes = new Map();
  const insertions = [];
  const children = [];
  const element = {
    tagName: tagName.toUpperCase(),
    nodeName: tagName.toUpperCase(),
    textContent: text,
    dataset: { ...dataset },
    attributes,
    insertions,
    children,
    parentElement: parent,
    parentNode: parent,
    hidden: false,
    style: {},
    get className() {
      return Array.from(classNames).join(" ");
    },
    set className(value) {
      classNames.clear();
      String(value).split(/\s+/u).filter(Boolean).forEach((name) => classNames.add(name));
    },
    classList: {
      add(...names) {
        names.forEach((name) => classNames.add(String(name)));
      },
      remove(...names) {
        names.forEach((name) => classNames.delete(String(name)));
      },
      contains(name) {
        return classNames.has(String(name));
      },
      toggle(name, force) {
        if (force === false) classNames.delete(String(name));
        else classNames.add(String(name));
      },
      toString() {
        return Array.from(classNames).join(" ");
      },
    },
    setAttribute(name, value = "") {
      const stringValue = String(value);
      attributes.set(name, stringValue);
      if (name === "class") stringValue.split(/\s+/u).filter(Boolean).forEach((item) => classNames.add(item));
      if (name.startsWith("data-")) {
        const key = name.slice(5).replace(/-([a-z])/gu, (_match, char) => char.toUpperCase());
        element.dataset[key] = stringValue;
      }
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    hasAttribute(name) {
      return attributes.has(name);
    },
    matches(selector) {
      const normalized = normalizeCss(selector);
      if (normalized === ".field-help-trigger") return classNames.has("field-help-trigger");
      if (normalized === ".sr-only") return classNames.has("sr-only");
      if (normalized.includes("[data-field-help]")) return attributes.has("data-field-help");
      if (normalized.split(",").some((part) => part.trim() === tagName.toLowerCase())) return true;
      return false;
    },
    closest(selector) {
      let current = element;
      while (current) {
        if (current.matches?.(selector)) return current;
        current = current.parentElement;
      }
      return null;
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    cloneNode() {
      return makeFakeElement(tagName, { text, dataset: { ...element.dataset }, allElements });
    },
    insertAdjacentHTML(position, markup) {
      insertions.push({ position, markup });
    },
    appendChild(child) {
      children.push(child);
      child.parentElement = element;
      child.parentNode = element;
      return child;
    },
    insertBefore(child, beforeNode) {
      const existingIndex = children.indexOf(child);
      if (existingIndex >= 0) children.splice(existingIndex, 1);
      const index = children.indexOf(beforeNode);
      if (index >= 0) children.splice(index, 0, child);
      else children.push(child);
      child.parentElement = element;
      child.parentNode = element;
      return child;
    },
    append(...nodes) {
      nodes.forEach((node) => element.appendChild(node));
    },
    prepend(...nodes) {
      nodes.reverse().forEach((node) => {
        children.unshift(node);
        node.parentElement = element;
        node.parentNode = element;
      });
    },
    before(...nodes) {
      insertions.push({ position: "beforebegin", markup: nodes.map(String).join("") });
    },
    after(...nodes) {
      insertions.push({ position: "afterend", markup: nodes.map(String).join("") });
    },
    toString() {
      return `<${tagName} class="${Array.from(classNames).join(" ")}">`;
    },
  };
  element.className = className;
  if (parent) parent.children.push(element);
  allElements.push(element);
  return element;
}

function fakeHydrationResult({
  headingDataset = { conceptHelp: "model_graph" },
  headingClass = "",
  headingParentClass = "",
} = {}) {
  const allElements = [];
  const root = makeFakeElement("section", { allElements });
  const headingParent = makeFakeElement("div", {
    allElements,
    className: headingParentClass,
    parent: root,
  });
  const heading = makeFakeElement("h2", {
    text: "模型语义组件图（Model Graph）",
    dataset: headingDataset,
    className: headingClass,
    parent: headingParent,
    allElements,
  });
  const inlineTarget = makeFakeElement("span", {
    text: "通用计算（GPU）",
    dataset: { conceptHelp: "gpu" },
    parent: root,
    allElements,
  });
  const document = {
    body: makeFakeElement("body", { allElements }),
    documentElement: { clientWidth: 1280, clientHeight: 720 },
    createElement(tagName) {
      return makeFakeElement(tagName, { allElements });
    },
  };
  const queryAll = (selector) => {
    if (selector === "[data-concept-help]") {
      return allElements.filter((item) => item.dataset.conceptHelp);
    }
    if (selector.includes("h1, h2, h3")) return [heading, inlineTarget];
    if (selector.includes("[data-field-help]")) {
      return allElements.filter((item) => item.attributes.has("data-field-help"));
    }
    if (selector.includes(".field-help-popover")) return [];
    return [];
  };
  const conceptHelpSource = sourceBetweenFunctions("conceptHelpSections", "closeFieldHelp");
  const hydrateConceptHelp = Function("fixtures", `
    const { document, queryAll } = fixtures;
    const $ = () => null;
    const $$ = queryAll;
    const CONCEPT_HELP = Object.freeze({ model_graph: "Model Graph", gpu: "GPU" });
    const CONCEPT_HELP_LABEL_BINDING_MAP = new Map();
    const CONCEPT_TERM_PATTERNS = [
      ["model_graph", /模型语义组件图|Model Graph/iu],
      ["gpu", /\bGPU\b|图形处理器/iu],
    ];
    let fieldHelpSerial = 0;
    let fieldHelpPortal = null;
    let activeFieldHelp = null;
    function bindFieldHelp() {}
    function normalizedConceptHelpLabel(value) { return String(value || "").trim().toLowerCase(); }
    function uiText(zh, en, replacements = {}) {
      return String(en || zh).replace(/\\{([^}]+)\\}/gu, (_match, key) => String(replacements[key] ?? ""));
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/gu, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }
    ${conceptHelpSource}
    conceptHelpSections = function conceptHelpSectionsForTest(helpKey) {
      return [{ title: "Definition", body: String(helpKey) }];
    };
    return hydrateConceptHelp;
  `)({ document, queryAll });

  hydrateConceptHelp(root);
  return { allElements, heading, headingParent };
}

function insertedMarkup(element) {
  return element.insertions.map((item) => item.markup).join("\n");
}

test("concept-help headings keep native heading semantics and delegate the trigger", () => {
  const headingTags = openingHeadingTagsWithConceptHelp();
  assert.ok(headingTags.length > 0, "fixture must include h1-h6 data-concept-help targets");
  for (const tag of headingTags) {
    assert.doesNotMatch(tag, /\brole\s*=\s*["'][^"']*button/iu, `${tag} must not be authored as a button`);
    assert.doesNotMatch(tag, /\bdata-field-help\b/iu, `${tag} must not be authored as the field-help control`);
    assert.doesNotMatch(tag, /\bfield-help-trigger\b/iu, `${tag} must not carry trigger styling directly`);
    assert.doesNotMatch(tag, /\baria-expanded\b/iu, `${tag} must not own interactive expanded state`);
    assert.doesNotMatch(tag, /\btabindex\s*=/iu, `${tag} must not become a focusable control`);
  }

  const conceptHelpHelpers = sourceBetweenFunctions("conceptHelpSections", "closeFieldHelp");
  assert.match(conceptHelpHelpers, /concept-help-heading-trigger/u, "heading hydration must create a dedicated heading trigger");

  const { allElements, heading, headingParent } = fakeHydrationResult();
  assert.equal(heading.attributes.get("role"), undefined, "hydration must not replace native heading semantics with role=button");
  assert.equal(heading.attributes.has("data-field-help"), false, "hydration must not make the heading itself the help control");
  assert.equal(heading.attributes.has("tabindex"), false, "hydration must not make the heading itself focusable");
  assert.equal(heading.classList.contains("field-help-trigger"), false, "hydration must not put trigger behavior on the heading");
  assert.equal(heading.attributes.has("aria-expanded"), false, "expanded state belongs to the separate trigger");

  const headingInnerMarkup = heading.insertions
    .filter((item) => ["afterbegin", "beforeend"].includes(item.position))
    .map((item) => item.markup)
    .join("\n");
  assert.doesNotMatch(
    headingInnerMarkup,
    /class=["'][^"']*\bfield-help-popover\b|role=["']tooltip["']/u,
    "the hidden tooltip must not be inserted inside the heading element",
  );

  const generatedMarkup = [insertedMarkup(heading), insertedMarkup(headingParent)].join("\n");
  const generatedTrigger = allElements.some((item) => item.classList.contains("concept-help-heading-trigger"))
    || /concept-help-heading-trigger/u.test(generatedMarkup);
  assert.ok(generatedTrigger, "hydration must delegate heading interactivity to .concept-help-heading-trigger");
});

test("automatic concept-help matching leaves an sr-only canvas title untouched", () => {
  const { allElements, heading, headingParent } = fakeHydrationResult({
    headingDataset: {},
    headingClass: "sr-only",
  });

  assert.equal(heading.dataset.conceptHelp, undefined, "sr-only titles must not receive an inferred concept key");
  assert.equal(heading.parentElement, headingParent, "automatic matching must preserve the original parent");
  assert.ok(headingParent.children.includes(heading), "the original title must remain in its parent");
  assert.equal(heading.attributes.has("data-field-help"), false);
  assert.equal(heading.attributes.has("role"), false);
  assert.equal(heading.classList.contains("field-help-trigger"), false);
  assert.equal(allElements.some((item) => item.classList.contains("concept-help-heading-shell")), false);
});

test("explicit concept-help matching leaves a title inside an sr-only ancestor untouched", () => {
  const { allElements, heading, headingParent } = fakeHydrationResult({
    headingParentClass: "sr-only",
  });

  assert.equal(heading.dataset.conceptHelp, "model_graph", "explicit metadata must remain intact");
  assert.equal(heading.parentElement, headingParent, "sr-only ancestors must preserve the original parent");
  assert.ok(headingParent.children.includes(heading), "the original title must remain in its parent");
  assert.equal(heading.attributes.has("data-field-help"), false);
  assert.equal(heading.attributes.has("role"), false);
  assert.equal(heading.classList.contains("field-help-trigger"), false);
  assert.equal(allElements.some((item) => item.classList.contains("concept-help-heading-shell")), false);
});

test("<=1280 responsive rules stack cross-page headers and wrap action controls", () => {
  const responsive = mediaBlockText(
    (condition) => conditionHasMaxWidth(condition, 1280),
    "<=1280 responsive",
  );

  assertDeclaration(responsive, ["view-head"], "flex-direction", /^column\b/u, "cross-page view heads");
  assertDeclaration(responsive, ["view-head"], "align-items", /^stretch\b/u, "cross-page view heads");

  for (const className of ["view-head-actions", "view-head-meta", "results-actions"]) {
    assertSomeDeclaration(
      responsive,
      [className],
      ["width", "flex", "flex-basis"],
      /(?:100%|1\s+1\s+100%)/u,
      `.${className} should take its own row`,
    );
    assertDeclarationInCascade(responsive, styles, [className], "flex-wrap", /^wrap\b/u, `.${className}`);
    assertDeclaration(responsive, [className, "button"], "min-width", /^0\b/u, `.${className} buttons`);
    assertDeclaration(responsive, [className, "button"], "max-width", /^100%$/u, `.${className} buttons`);
    assertDeclaration(responsive, [className, "button"], "white-space", /^normal\b/u, `.${className} buttons`);
  }

  assertDeclaration(responsive, ["topbar"], "grid-template-columns", /minmax\(0,\s*1fr\)/u, "topbar");
  assertDeclaration(responsive, ["top-actions"], "flex-wrap", /^wrap\b/u, ".top-actions");
  assertDeclaration(responsive, ["top-actions"], "min-width", /^0\b/u, ".top-actions");
  assertDeclaration(responsive, ["top-actions", "button"], "min-width", /^0\b/u, ".top-actions buttons");
  assertDeclaration(responsive, ["top-actions", "button"], "max-width", /^100%$/u, ".top-actions buttons");
  assertDeclaration(responsive, ["top-actions", "button"], "white-space", /^normal\b/u, ".top-actions buttons");
  assertSomeDeclaration(
    responsive,
    ["top-actions", "button"],
    ["overflow-wrap", "word-break"],
    /(?:anywhere|break-word)/u,
    ".top-actions buttons",
  );
});

test("1061-1280 architecture uses two columns with inspector below while <=1060 remains one column", () => {
  const mediumArchitecture = mediaBlockText(
    (condition) => conditionHasMinWidth(condition, 1061) && conditionHasMaxWidth(condition, 1280),
    "1061-1280 architecture",
  );
  const mediumColumns = lastDeclarationForClasses(
    mediumArchitecture,
    ["architecture-grid"],
    "grid-template-columns",
    "medium architecture grid",
  );
  assert.equal(
    gridTrackCount(mediumColumns),
    2,
    `1061-1280 architecture grid must have two top-level columns, got ${mediumColumns}`,
  );
  assertDeclaration(mediumArchitecture, ["inspector"], "grid-column", /^1\s*\/\s*-1$/u, "medium inspector");
  assertDeclaration(mediumArchitecture, ["inspector"], "border-top", /1px\s+solid\s+var\(--line\)/u, "medium inspector");
  assertDeclaration(mediumArchitecture, ["inspector"], "border-left", /^0$/u, "medium inspector");

  const narrowArchitecture = mediaBlockText(
    (condition) => conditionHasMaxWidth(condition, 1060),
    "<=1060 architecture",
  );
  const narrowColumns = lastDeclarationForClasses(
    narrowArchitecture,
    ["architecture-grid"],
    "grid-template-columns",
    "narrow architecture grid",
  );
  assert.equal(
    gridTrackCount(narrowColumns),
    1,
    `<=1060 architecture grid must remain one column, got ${narrowColumns}`,
  );
  assert.match(narrowColumns, /minmax\(0,\s*1fr\)/u);
  assertDeclaration(narrowArchitecture, ["inspector"], "border-top", /1px\s+solid\s+var\(--line\)/u, "narrow inspector");
  assertDeclaration(narrowArchitecture, ["inspector"], "border-left", /^0$/u, "narrow inspector");
});
