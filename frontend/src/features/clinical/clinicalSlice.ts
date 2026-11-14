import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ClinicalState {
  messages: ChatMessage[];
  isLoading: boolean;
}

const initialState: ClinicalState = {
  messages: [],
  isLoading: false,
};

const clinicalSlice = createSlice({
  name: 'clinical',
  initialState,
  reducers: {
    addMessage(state, action: PayloadAction<ChatMessage>) {
      state.messages.push(action.payload);
    },
    setLoading(state, action: PayloadAction<boolean>) {
      state.isLoading = action.payload;
    },
    clearMessages(state) {
      state.messages = [];
    },
  },
});

export const { addMessage, setLoading, clearMessages } = clinicalSlice.actions;
export default clinicalSlice.reducer;
