import { NextResponse } from "next/server";
import { TICKETS } from "../../../lib/tickets";

export async function GET() {
  return NextResponse.json({ count: TICKETS.length, tickets: TICKETS });
}
