import { Routes, Route } from "react-router-dom";
import { I18nProvider } from "./i18n";
import ChatPage from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";

export default function App() {
  return (
    <I18nProvider>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </I18nProvider>
  );
}
