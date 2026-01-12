import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Moon, Sun } from "lucide-react";
import WeeklyPostMortem from "@/components/admin/WeeklyPostMortem";
import DailyBiasPanel from "@/components/admin/DailyBiasPanel";
import WhitelistPanel from "@/components/admin/WhitelistPanel";
import CurrencyPairsPanel from "@/components/admin/CurrencyPairsPanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Admin = () => {
  const navigate = useNavigate();
  const [isDark, setIsDark] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        navigate("/auth");
        return;
      }
      // For now, any authenticated user can access admin
      // You can add role-based access later
      setIsAdmin(true);
      setLoading(false);
    };
    checkAuth();
  }, [navigate]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground">Access denied</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/dashboard")}
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <h1 className="text-xl font-bold text-foreground">Admin Panel</h1>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsDark(!isDark)}
          >
            {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs defaultValue="pairs" className="w-full">
          <TabsList className="grid w-full max-w-2xl grid-cols-4 mb-6">
            <TabsTrigger value="pairs">Currency Pairs</TabsTrigger>
            <TabsTrigger value="weekly">Weekly Post-Mortem</TabsTrigger>
            <TabsTrigger value="daily">Daily Bias</TabsTrigger>
            <TabsTrigger value="whitelist">Whitelist</TabsTrigger>
          </TabsList>
          
          <TabsContent value="pairs">
            <CurrencyPairsPanel />
          </TabsContent>
          
          <TabsContent value="weekly">
            <WeeklyPostMortem />
          </TabsContent>
          
          <TabsContent value="daily">
            <DailyBiasPanel />
          </TabsContent>
          
          <TabsContent value="whitelist">
            <WhitelistPanel />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Admin;
