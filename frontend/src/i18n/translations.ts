export type Locale = "he" | "en";

export const translations = {
  he: {
    // Chat sidebar
    appName: "Electricity RAG",
    newChat: "צ'אט חדש",
    adminPanel: "פאנל ניהול",
    closeSidebar: "סגור תפריט",

    // Chat empty state
    emptyTitle: "עוזר ידע חשמל",
    emptyDescription: "שאלו אותי כל שאלה על חשמל, מערכות חשמל, תקנות ועוד.",

    // Chat input
    inputPlaceholder: "שאלו על חשמל...",
    disclaimer: "מערכת זו היא כלי עזר לאיתור מידע בתקנות החשמל ואינה מחליפה שיקול דעת של מהנדס חשמל מוסמך.",
    errorMessage: "מצטער, משהו השתבש. נסו שוב.",
    sourcesLabel: "מקורות",

    // Admin login
    adminAccess: "גישת מנהל",
    enterPassword: "הזינו את סיסמת המנהל לניהול מסמכים.",
    password: "סיסמה",
    signIn: "כניסה",
    backToChat: "חזרה לצ'אט",
    incorrectPassword: "סיסמה שגויה",

    // Admin panel
    adminPanelTitle: "פאנל ניהול",
    vectorStore: "מאגר וקטורים",
    active: "פעיל",
    empty: "ריק",
    documents: "מסמכים",
    files: "קבצים",
    uploadDocuments: "העלאת מסמכים",
    uploadDescription: "העלו קבצי PDF, DOCX, TXT, XLSX או CSV. כל קובץ יעובד על ידי LLM ליצירת מסמך מותאם למאגר הידע, ולאחר מכן יאונדקס לאחזור.",
    processingFile: "מעבד קובץ עם LLM... זה עשוי לקחת דקה.",
    clickToSelect: "לחצו לבחירת קובץ",
    uploadedFiles: "קבצים שהועלו",
    noFiles: "עדיין לא הועלו קבצים.",
    deleteConfirm: (name: string) => `למחוק את "${name}" ולבנות מחדש את מאגר הידע?`,
    uploadSuccess: (name: string, chunks: number) => `"${name}" עובד בהצלחה. ${chunks} חלקים אונדקסו.`,
    uploadFailed: "ההעלאה נכשלה",

    // Q&A audit log
    qaLogTitle: "יומן שאלות ותשובות",
    qaLogDescription: "היסטוריית שאלות אחרונות, המקורות שנשלפו לכל שאלה והתשובה שנוצרה. השתמשו ביומן לסימון תשובות שגויות או לא מדויקות.",
    qaLogCount: (n: number) => `${n} רשומות ביומן`,
    qaLogEmpty: "עדיין אין רשומות ביומן.",
    qaLogRefused: "לא נענתה",
    qaLogRefreshed: "רענן",
    qaLogQuestion: "שאלה",
    qaLogAnswer: "תשובה",
    qaLogChunks: "קטעים שנשלפו",
    qaLogAltQueries: "שאילתות חלופיות",
    qaLogReferencedRegs: "תקנות שאוזכרו",

    // Language toggle
    langToggle: "EN",
  },
  en: {
    appName: "Electricity RAG",
    newChat: "New Chat",
    adminPanel: "Admin Panel",
    closeSidebar: "Close sidebar",

    emptyTitle: "Electricity Knowledge Assistant",
    emptyDescription: "Ask me anything about electricity, power systems, regulations, and more.",

    inputPlaceholder: "Ask about electricity...",
    disclaimer: "This system is an aid for finding information in Israeli electricity regulations and does not replace the judgement of a licensed electrical engineer.",
    errorMessage: "Sorry, something went wrong. Please try again.",
    sourcesLabel: "Sources",

    adminAccess: "Admin Access",
    enterPassword: "Enter the admin password to manage documents.",
    password: "Password",
    signIn: "Sign In",
    backToChat: "Back to chat",
    incorrectPassword: "Incorrect password",

    adminPanelTitle: "Admin Panel",
    vectorStore: "Vector Store",
    active: "Active",
    empty: "Empty",
    documents: "Documents",
    files: "files",
    uploadDocuments: "Upload Documents",
    uploadDescription: "Upload PDF, DOCX, TXT, XLSX, or CSV files. Each file will be processed by an LLM to create an optimized knowledge base document, then indexed for retrieval.",
    processingFile: "Processing file with LLM... This may take a minute.",
    clickToSelect: "Click to select a file",
    uploadedFiles: "Uploaded Files",
    noFiles: "No files uploaded yet.",
    deleteConfirm: (name: string) => `Delete "${name}" and rebuild the knowledge base?`,
    uploadSuccess: (name: string, chunks: number) => `"${name}" processed successfully. ${chunks} chunks indexed.`,
    uploadFailed: "Upload failed",

    langToggle: "עב",
  },
} as const;

export type TranslationKeys = typeof translations.en;
