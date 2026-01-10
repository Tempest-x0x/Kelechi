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
          opportunity_id: string | null
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
          opportunity_id?: string | null
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
          opportunity_id?: string | null
          pattern_context?: Json | null
          prediction_id?: string | null
          success_factors?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "prediction_learnings_opportunity_id_fkey"
            columns: ["opportunity_id"]
            isOneToOne: false
            referencedRelation: "trading_opportunities"
            referencedColumns: ["id"]
          },
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
      profiles: {
        Row: {
          created_at: string
          email: string
          full_name: string | null
          id: string
        }
        Insert: {
          created_at?: string
          email: string
          full_name?: string | null
          id: string
        }
        Update: {
          created_at?: string
          email?: string
          full_name?: string | null
          id?: string
        }
        Relationships: []
      }
      subscriptions: {
        Row: {
          amount: number
          created_at: string
          current_period_end: string | null
          current_period_start: string
          id: string
          paystack_customer_code: string | null
          paystack_reference: string | null
          paystack_subscription_code: string | null
          plan_type: string
          status: string
          updated_at: string
          user_id: string
        }
        Insert: {
          amount: number
          created_at?: string
          current_period_end?: string | null
          current_period_start?: string
          id?: string
          paystack_customer_code?: string | null
          paystack_reference?: string | null
          paystack_subscription_code?: string | null
          plan_type: string
          status?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          amount?: number
          created_at?: string
          current_period_end?: string | null
          current_period_start?: string
          id?: string
          paystack_customer_code?: string | null
          paystack_reference?: string | null
          paystack_subscription_code?: string | null
          plan_type?: string
          status?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "subscriptions_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      trading_opportunities: {
        Row: {
          ai_learning_id: string | null
          confidence: number
          created_at: string | null
          current_price: number
          entry_price: number
          evaluated_at: string | null
          expires_at: string
          id: string
          notification_sent_at: string | null
          outcome: string | null
          pattern_stats: Json | null
          patterns_detected: Json | null
          reasoning: string | null
          signal_type: string
          status: string | null
          stop_loss: number | null
          symbol: string | null
          take_profit_1: number | null
          take_profit_2: number | null
          technical_indicators: Json | null
          triggered_at: string | null
        }
        Insert: {
          ai_learning_id?: string | null
          confidence: number
          created_at?: string | null
          current_price: number
          entry_price: number
          evaluated_at?: string | null
          expires_at: string
          id?: string
          notification_sent_at?: string | null
          outcome?: string | null
          pattern_stats?: Json | null
          patterns_detected?: Json | null
          reasoning?: string | null
          signal_type: string
          status?: string | null
          stop_loss?: number | null
          symbol?: string | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          triggered_at?: string | null
        }
        Update: {
          ai_learning_id?: string | null
          confidence?: number
          created_at?: string | null
          current_price?: number
          entry_price?: number
          evaluated_at?: string | null
          expires_at?: string
          id?: string
          notification_sent_at?: string | null
          outcome?: string | null
          pattern_stats?: Json | null
          patterns_detected?: Json | null
          reasoning?: string | null
          signal_type?: string
          status?: string | null
          stop_loss?: number | null
          symbol?: string | null
          take_profit_1?: number | null
          take_profit_2?: number | null
          technical_indicators?: Json | null
          triggered_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "trading_opportunities_ai_learning_id_fkey"
            columns: ["ai_learning_id"]
            isOneToOne: false
            referencedRelation: "prediction_learnings"
            referencedColumns: ["id"]
          },
        ]
      }
      user_roles: {
        Row: {
          created_at: string | null
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
      whitelisted_emails: {
        Row: {
          added_by: string | null
          created_at: string | null
          email: string
          expires_at: string | null
          id: string
          reason: string | null
        }
        Insert: {
          added_by?: string | null
          created_at?: string | null
          email: string
          expires_at?: string | null
          id?: string
          reason?: string | null
        }
        Update: {
          added_by?: string | null
          created_at?: string | null
          email?: string
          expires_at?: string | null
          id?: string
          reason?: string | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
    }
    Enums: {
      app_role: "admin" | "user"
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
    Enums: {
      app_role: ["admin", "user"],
    },
  },
} as const
