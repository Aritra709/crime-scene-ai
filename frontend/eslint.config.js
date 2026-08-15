export default [
  { ignores: ["dist", "node_modules"] },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
      ecmaFeatures: { jsx: true },
      parserOptions: { project: "./tsconfig.json" },
    },
    settings: { react: { version: "18.3" } },
    plugins: {
      react: await import("eslint-plugin-react"),
      "react-hooks": await import("eslint-plugin-react-hooks"),
      "@typescript-eslint": await import("@typescript-eslint/eslint-plugin"),
    },
    rules: {
      "react/react-in-jsx-scope": "off",
      "react/prop-types": "off",
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "react-hooks/exhaustive-deps": "warn",
    },
  },
];