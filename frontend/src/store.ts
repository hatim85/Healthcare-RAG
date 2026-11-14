import { configureStore } from "@reduxjs/toolkit";
import { setupListeners } from "@reduxjs/toolkit/query";
import { ragApi } from "@/app/api";
import authReducer from "@/features/auth/authSlice";
import clinicalReducer from "@/features/clinical/clinicalSlice";

export const store = configureStore({
  reducer: {
    [ragApi.reducerPath]: ragApi.reducer,
    auth: authReducer,
    clinical: clinicalReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(ragApi.middleware),
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

