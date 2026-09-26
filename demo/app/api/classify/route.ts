import { NextResponse } from "next/server";
import { buildPrompt, classifyTicket, type Ticket } from "../../../lib/tell";

export async function POST(req: Request) {
  let ticket: Ticket;
  try {
    ticket = (await req.json()) as Ticket;
  } catch {
    return NextResponse.json({ label: null, status: "invalid_output" }, { status: 400 });
  }
  if (!ticket || !ticket.state || !ticket.question || !Array.isArray(ticket.options)) {
    return NextResponse.json({ label: null, status: "invalid_output" }, { status: 400 });
  }
  const result = classifyTicket(ticket);
  return NextResponse.json({ ...result, prompt: buildPrompt(ticket) });
}
