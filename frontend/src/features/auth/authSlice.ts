import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { AuthToken } from "@/types";

export interface AuthState {
  token: string | null;
  username: string | null;
  role: string | null;
  tenant_id: string | null;
}

const initialState: AuthState = {
  token: localStorage.getItem("token"),
  username: localStorage.getItem("username"),
  role: localStorage.getItem("role"),
  tenant_id: localStorage.getItem("tenant_id"),
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    setCredentials(state, action: PayloadAction<AuthToken>) {
      const { access_token, username, role, tenant_id } = action.payload;
      state.token = access_token;
      state.username = username;
      state.role = role;
      state.tenant_id = tenant_id;

      localStorage.setItem("token", access_token);
      localStorage.setItem("username", username);
      localStorage.setItem("role", role);
      localStorage.setItem("tenant_id", tenant_id);
    },

    logout(state) {
      state.token = null;
      state.username = null;
      state.role = null;
      state.tenant_id = null;

      localStorage.removeItem("token");
      localStorage.removeItem("username");
      localStorage.removeItem("role");
      localStorage.removeItem("tenant_id");
    },
  },
});

export const { setCredentials, logout } = authSlice.actions;
export default authSlice.reducer;
