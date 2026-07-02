import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { translations, type Locale, type TranslationKeys } from "./translations";

interface I18nContextType {
  locale: Locale;
  t: TranslationKeys;
  toggleLocale: () => void;
}

const I18nContext = createContext<I18nContextType | null>(null);

const STORAGE_KEY = "electricity_rag_locale";

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return (saved === "en" || saved === "he") ? saved : "he";
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, locale);
    document.documentElement.dir = locale === "he" ? "rtl" : "ltr";
    document.documentElement.lang = locale;
  }, [locale]);

  const toggleLocale = useCallback(() => {
    setLocale((prev) => (prev === "he" ? "en" : "he"));
  }, []);

  const t = translations[locale];

  return (
    <I18nContext.Provider value={{ locale, t, toggleLocale }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}
