import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { Loader2, Plus, Trash2, Mail, Calendar } from "lucide-react";
import { format } from "date-fns";

interface WhitelistedEmail {
  id: string;
  email: string;
  reason: string | null;
  created_at: string;
  expires_at: string | null;
}

const WhitelistPanel = () => {
  const { toast } = useToast();
  const [emails, setEmails] = useState<WhitelistedEmail[]>([]);
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  
  const [newEmail, setNewEmail] = useState("");
  const [newReason, setNewReason] = useState("");

  const fetchEmails = async () => {
    try {
      const { data, error } = await supabase
        .from("whitelisted_emails")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      setEmails(data || []);
    } catch (error) {
      console.error("Error fetching whitelist:", error);
      toast({
        title: "Error",
        description: "Failed to load whitelisted emails",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEmails();
  }, []);

  const handleAdd = async () => {
    if (!newEmail.trim()) {
      toast({
        title: "Error",
        description: "Please enter an email address",
        variant: "destructive",
      });
      return;
    }

    setAdding(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      const { error } = await supabase
        .from("whitelisted_emails")
        .insert({
          email: newEmail.trim().toLowerCase(),
          reason: newReason.trim() || null,
          added_by: user?.id,
        });

      if (error) {
        if (error.code === "23505") {
          throw new Error("This email is already whitelisted");
        }
        throw error;
      }

      toast({
        title: "Email Added",
        description: `${newEmail} has been whitelisted`,
      });

      setNewEmail("");
      setNewReason("");
      fetchEmails();
    } catch (error: any) {
      console.error("Error adding email:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to add email",
        variant: "destructive",
      });
    } finally {
      setAdding(false);
    }
  };

  const handleDelete = async (id: string, email: string) => {
    setDeletingId(id);
    try {
      const { error } = await supabase
        .from("whitelisted_emails")
        .delete()
        .eq("id", id);

      if (error) throw error;

      toast({
        title: "Email Removed",
        description: `${email} has been removed from whitelist`,
      });

      fetchEmails();
    } catch (error) {
      console.error("Error deleting email:", error);
      toast({
        title: "Error",
        description: "Failed to remove email",
        variant: "destructive",
      });
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Add New Email */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mail className="h-5 w-5" />
            Add Email to Whitelist
          </CardTitle>
          <CardDescription>
            Whitelisted emails can access the app for free without a subscription
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <Input
                id="email"
                type="email"
                placeholder="user@example.com"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="reason">Reason (optional)</Label>
              <Input
                id="reason"
                placeholder="e.g., Beta tester, Partner"
                value={newReason}
                onChange={(e) => setNewReason(e.target.value)}
              />
            </div>
          </div>
          <Button onClick={handleAdd} disabled={adding}>
            {adding ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Adding...
              </>
            ) : (
              <>
                <Plus className="mr-2 h-4 w-4" />
                Add to Whitelist
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Whitelisted Emails List */}
      <Card>
        <CardHeader>
          <CardTitle>Whitelisted Emails ({emails.length})</CardTitle>
          <CardDescription>
            Users with these emails can access the full dashboard without payment
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : emails.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No emails whitelisted yet
            </div>
          ) : (
            <div className="space-y-3">
              {emails.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between p-3 rounded-lg border bg-card"
                >
                  <div className="space-y-1">
                    <div className="font-medium">{item.email}</div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      {item.reason && <span>{item.reason}</span>}
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        Added {format(new Date(item.created_at), "MMM d, yyyy")}
                      </span>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDelete(item.id, item.email)}
                    disabled={deletingId === item.id}
                  >
                    {deletingId === item.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4 text-destructive" />
                    )}
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default WhitelistPanel;
