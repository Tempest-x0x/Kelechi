export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      pattern_statistics: {
        Row: {
          avg_pips_12h: number | null
          avg_pips_24h: number | null
          avg_pips_48h: number | null
          avg_pips_4h: number | null
          created_at: string | null
          data_end_date: string | null
          data_start_date: string | null
          id: string
          occurrences: number
          pattern_name: string
          sample_size: number | null
          signal_type: string
          win_rate_12h: number | null
          win_rate_24h: number | null
          win_rate_48h: number | null
          win_rate_4h: number | null
        }
        Insert: {
          avg_pips_12h?: number | null
          avg_pips_24h?: number | null
          avg_pips_48h?: number | null
          avg_pips_4h?: number | null
          created_at?: string | null
          data_end_date?: string | null
          data_start_date?: string | null
          id?: string
          occurrences: number
          pattern_name: string
          sample_size?: number | null
          signal_type: string
          win_rate_12h?: number | null
          win_rate_24h?: number | null
          win_rate_48h?: number | null
          win_rate_4h?: number | null
        }
        Update: {
          avg_pips_12h?: number | null
          avg_pips_24h?: number | null
          avg_pips_48h?: number | null
          avg_pips_4h?: number | null
          created_at?: string | null
          data_end_date?: string | null
          data_start_date?: string | null
          id?: string
          occurrences?: number
          pattern_name?: string
          sample_size?: number | null
          signal_type?: string
          win_rate_12h?: number | null
          win_rate_24h?: number | null
          win_rate_48h?: number | null
          win_rate_4h?: number | null
        }
        Relationships: []
      }
      prediction_learnings: {
        Row: {
          created_at: string
          failure_reason: string | null
          id: string
          lesson_extracted: string
          market_conditions: Json | null
          pattern_context: Json | null
          prediction_id: string | null
          success_factors: string | null
        }
        Insert: {
          created_at?: string
          failure_reason?: string | null
          id?: string
          lesson_extracted: string
          market_conditions?: Json | null
          pattern_context?: Json | null
          prediction_id?: string | null
          success_factors?: string | null
        }
        Update: {
          created_at?: string
          failure_reason?: string | null
          id?: string
          lesson_extracted?: string
          market_conditions?: Json | null
          pattern_context?: Json | null
          prediction_id?: string | null
          success_factors?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "prediction_learnings_prediction_id_fkey"
            columns: ["prediction_id"]
            isOneToOne: false
            referencedRelation: "predictions"
            referencedColumns: ["id"]
          },
        ]
      }
      predictions: {
        Row: {
          confidence: number
          created_at: string
          current_price_at_prediction: number
          entry_price: number
          expires_at: string
          id: string
          outcome: string | null
          outcome_at: string | null
          outcome_price: number | null
          patterns_detected: Json | null
          reasoning: string | null
          sentiment_score: number | null
          signal_type: string
          stop_loss: number | null
          take_profit_1: number | null
          take_profit_2: number | null
          technical_indicators: Json | null
          trend_direction: string
          trend_strength: number | null
        }
        Insert: {
          confidence: number
          created_at?: string
          current_price_at_prediction: number
          entry_price: number
          expires_at?: string
          id?: string
          outcome?: string | null
          outcome_at?: string | null
          outcome_price?: number | null
          patterns_detected?: Json | null
          reasoning?: string | null
          sentiment_score?: number | null
          signal_type: string
          stop_loss?: number | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          trend_direction: string
          trend_strength?: number | null
        }
        Update: {
          confidence?: number
          created_at?: string
          current_price_at_prediction?: number
          entry_price?: number
          expires_at?: string
          id?: string
          outcome?: string | null
          outcome_at?: string | null
          outcome_price?: number | null
          patterns_detected?: Json | null
          reasoning?: string | null
          sentiment_score?: number | null
          signal_type?: string
          stop_loss?: number | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          trend_direction?: string
          trend_strength?: number | null
        }
        Relationships: []
      }
      price_history: {
        Row: {
          close: number
          high: number
          id: string
          low: number
          open: number
          symbol: string
          timeframe: string
          timestamp: string
          volume: number | null
        }
        Insert: {
          close: number
          high: number
          id?: string
          low: number
          open: number
          symbol?: string
          timeframe?: string
          timestamp: string
          volume?: number | null
        }
        Update: {
          close?: number
          high?: number
          id?: string
          low?: number
          open?: number
          symbol?: string
          timeframe?: string
          timestamp?: string
          volume?: number | null
        }
        Relationships: []
      }
      trading_opportunities: {
        Row: {
          confidence: number
          created_at: string | null
          current_price: number
          entry_price: number
          expires_at: string
          id: string
          outcome: string | null
          pattern_stats: Json | null
          patterns_detected: Json | null
          reasoning: string | null
          signal_type: string
          status: string | null
          stop_loss: number | null
          take_profit_1: number | null
          take_profit_2: number | null
          technical_indicators: Json | null
          triggered_at: string | null
        }
        Insert: {
          confidence: number
          created_at?: string | null
          current_price: number
          entry_price: number
          expires_at: string
          id?: string
          outcome?: string | null
          pattern_stats?: Json | null
          patterns_detected?: Json | null
          reasoning?: string | null
          signal_type: string
          status?: string | null
          stop_loss?: number | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          triggered_at?: string | null
        }
        Update: {
          confidence?: number
          created_at?: string | null
          current_price?: number
          entry_price?: number
          expires_at?: string
          id?: string
          outcome?: string | null
          pattern_stats?: Json | null
          patterns_detected?: Json | null
          reasoning?: string | null
          signal_type?: string
          status?: string | null
          stop_loss?: number | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          triggered_at?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
