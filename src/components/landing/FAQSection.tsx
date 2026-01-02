import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

const faqs = [
  {
    question: 'Is this a signal service?',
    answer: 'No. ForexTell AI is a decision engine, not a signal service. We provide directional bias and key levels, not entry/exit signals. You decide when and how to trade based on our analysis.',
  },
  {
    question: 'What currency pairs do you cover?',
    answer: 'We focus exclusively on EUR/USD. This laser focus allows us to provide deeper analysis and better accuracy than services that try to cover everything.',
  },
  {
    question: 'What if the bias is wrong?',
    answer: 'Every bias comes with an invalidation level. If price hits that level, you know the bias was wrong and can exit with a controlled loss. No hoping, no praying—just rules.',
  },
  {
    question: 'How is this different from other Forex services?',
    answer: 'Most services sell you signals and promise unrealistic returns. We provide institutional-style analysis: directional bias, invalidation levels, and target zones. We show our track record publicly. No hype, just data.',
  },
  {
    question: 'Can I cancel anytime?',
    answer: 'Yes. Monthly subscriptions can be cancelled at any time. You\'ll retain access until the end of your billing period.',
  },
  {
    question: 'Is my payment secure?',
    answer: 'Yes. All payments are processed securely through Paystack, a trusted payment provider used by major banks and businesses across Africa.',
  },
  {
    question: 'What\'s included in the Lifetime plan?',
    answer: 'Lifetime members get everything in the Funded Trader plan, forever. No recurring payments, all future features included, plus VIP support and early access to new currency pairs when we expand.',
  },
];

export const FAQSection = () => {
  return (
    <section className="py-20">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Frequently Asked Questions</h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Got questions? We've got answers.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <Accordion type="single" collapsible className="space-y-4">
            {faqs.map((faq, index) => (
              <AccordionItem 
                key={index} 
                value={`item-${index}`}
                className="bg-card border border-border rounded-lg px-6"
              >
                <AccordionTrigger className="text-left hover:no-underline">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </div>
    </section>
  );
};
