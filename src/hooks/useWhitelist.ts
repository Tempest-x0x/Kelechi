import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";

export const useWhitelist = () => {
  const [isWhitelisted, setIsWhitelisted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkWhitelist = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        
        if (!user?.email) {
          setIsLoading(false);
          return;
        }

        const { data, error } = await supabase
          .from("whitelisted_emails")
          .select("id, expires_at")
          .eq("email", user.email.toLowerCase())
          .maybeSingle();

        if (error) {
          console.error("Error checking whitelist:", error);
          setIsLoading(false);
          return;
        }

        if (data) {
          // Check if not expired
          if (data.expires_at) {
            const isValid = new Date(data.expires_at) > new Date();
            setIsWhitelisted(isValid);
          } else {
            // No expiry = always valid
            setIsWhitelisted(true);
          }
        } else {
          setIsWhitelisted(false);
        }
      } catch (err) {
        console.error("Whitelist check error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    checkWhitelist();

    // Re-check on auth state change
    const { data: { subscription } } = supabase.auth.onAuthStateChange(() => {
      checkWhitelist();
    });

    return () => subscription.unsubscribe();
  }, []);

  return { isWhitelisted, isLoading };
};
