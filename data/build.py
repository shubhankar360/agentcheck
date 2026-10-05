"""Generate world.json and tasks.json. Deterministic: run it and diff.

Every task states the *end state* a correct agent leaves behind, not the
path it takes, so any valid sequence of tool calls passes. Tasks are grouped
so the categories that test policy (expired window, over the escalation
limit, someone else's order, shipped orders) are as numerous as the ones
that test "do the obvious thing".
"""

import json
from pathlib import Path

TODAY = "2026-10-05"

CUSTOMERS = [
    ("C1", "Priya Nair", "priya.nair@example.com"), ("C2", "Tom Becker", "tom.becker@example.com"),
    ("C3", "Aisha Khan", "aisha.khan@example.com"), ("C4", "Diego Ramos", "diego.ramos@example.com"),
    ("C5", "Mei Lin", "mei.lin@example.com"), ("C6", "Sam Okafor", "sam.okafor@example.com"),
    ("C7", "Lena Fischer", "lena.fischer@example.com"), ("C8", "Omar Haddad", "omar.haddad@example.com"),
]
EMAIL = {cid: email for cid, _, email in CUSTOMERS}
ADDR = {"C1": "14 Park Lane, Pune 411001", "C2": "3 Mill Road, Cambridge CB1 2AD", "C3": "88 Canal St, Manchester M1 3HE",
        "C4": "Calle Mayor 5, 28013 Madrid", "C5": "9 Orchard Rd, Singapore 238823", "C6": "21 Allen Ave, Lagos 100271",
        "C7": "Torstrasse 40, 10119 Berlin", "C8": "Al Wasl Rd 12, Dubai"}

orders: list[dict] = []
tasks: list[dict] = []


def order(oid, cid, items, status, delivered=None):
    total = round(sum(i["qty"] * i["price"] for i in items), 2)
    orders.append({"id": oid, "customer_id": cid, "items": items, "total": total, "status": status,
                   "delivered_at": delivered, "address": ADDR[cid]})
    return total


def item(sku, name, price, qty=1):
    return {"sku": sku, "name": name, "qty": qty, "price": price}


def task(tid, category, intent, frm, message, expect):
    tasks.append({"id": tid, "category": category, "intent": intent, "from": frm, "message": message,
                  "today": TODAY, "expect": {"refunds": [], "tickets": [], "status": {}, "address": {}, **expect}})


def money(x):
    return [f"{x:.2f}", f"${x:g}" if x == int(x) else f"${x:.2f}"]


# -- eligible refunds --------------------------------------------------------
for tid, oid, cid, items, delivered, msg in [
    ("refund-01", "A1001", "C1", [item("MUG-01", "ceramic mug", 14.50, 2)], "2026-09-28",
     "Hi, order A1001 arrived with both mugs smashed. Can I get my money back?"),
    ("refund-02", "A1002", "C2", [item("LMP-02", "desk lamp", 89.99)], "2026-09-20",
     "The desk lamp from order A1002 doesn't turn on. Please refund it."),
    ("refund-03", "A1003", "C3", [item("HPH-03", "headphones", 59.00)], "2026-10-01",
     "Order A1003: the headphones are broken out of the box, I'd like a refund."),
    ("refund-04", "A1004", "C4", [item("BPK-04", "backpack", 120.00)], "2026-09-10",
     "The zip on my backpack (order A1004) broke the first week. Refund please."),
    ("refund-05", "A1005", "C5", [item("KTL-05", "kettle", 45.50)], "2026-09-30",
     "My kettle from A1005 leaks. I want a refund."),
    ("refund-06", "A1006", "C6", [item("NBK-06", "notebook", 8.00, 3)], "2026-09-25",
     "All three notebooks in order A1006 came water-damaged. Please refund the order."),
]:
    total = order(oid, cid, items, "delivered", delivered)
    task(tid, "refund", "refund", EMAIL[cid], msg,
         {"refunds": [{"order_id": oid, "amount": total}], "answer": [["refund"], money(total)]})

# -- partial refunds ---------------------------------------------------------
for tid, oid, cid, items, delivered, msg, amount in [
    ("partial-01", "A1101", "C7", [item("MUG-01", "ceramic mug", 14.50), item("TPT-11", "teapot", 38.00)], "2026-09-29",
     "In order A1101 the teapot arrived cracked; the mug is fine. Refund just the teapot please.", 38.00),
    ("partial-02", "A1102", "C8", [item("CHR-12", "chair", 149.00), item("CSH-12", "cushion", 25.00)], "2026-09-22",
     "The cushion from A1102 is the wrong colour, please refund the cushion. Keeping the chair.", 25.00),
    ("partial-03", "A1103", "C1", [item("CDL-13", "candle", 12.00, 2), item("VAS-13", "vase", 40.00)], "2026-09-27",
     "One of the two candles in A1103 was melted. Refund for that one candle only.", 12.00),
]:
    order(oid, cid, items, "delivered", delivered)
    task(tid, "refund-partial", "refund", EMAIL[cid], msg,
         {"refunds": [{"order_id": oid, "amount": amount}], "answer": [["refund"], money(amount)]})

# -- outside the 30-day window ------------------------------------------------
for tid, oid, cid, items, delivered, msg in [
    ("expired-01", "A1201", "C2", [item("SPK-21", "speaker", 75.00)], "2026-08-15",
     "The speaker from order A1201 stopped working. Refund please."),
    ("expired-02", "A1202", "C3", [item("BLN-22", "blender", 65.00)], "2026-08-28",
     "Order A1202 — the blender motor burnt out. I'd like my money back."),
    ("expired-03", "A1203", "C5", [item("JKT-23", "jacket", 110.00)], "2026-09-04",
     "The jacket from A1203 has a torn seam. Can you refund it?"),
]:
    order(oid, cid, items, "delivered", delivered)
    task(tid, "refund-expired", "refund", EMAIL[cid], msg,
         {"answer": [["30 days", "30-day", "thirty"]]})

# -- above the $200 direct-refund limit: escalate ------------------------------
for tid, oid, cid, items, delivered, msg in [
    ("escalate-01", "A1301", "C4", [item("MON-31", "monitor", 349.00)], "2026-09-26",
     "The monitor from order A1301 has dead pixels. Please refund me."),
    ("escalate-02", "A1302", "C6", [item("HLM-32", "bike helmet", 89.00), item("LGT-32", "bike lights", 129.00)], "2026-09-30",
     "Neither item in order A1302 is what I ordered. Refund the whole order please."),
    ("escalate-03", "A1303", "C8", [item("ESP-33", "espresso machine", 499.00)], "2026-10-02",
     "My espresso machine (A1303) arrived dented. I want a full refund."),
]:
    order(oid, cid, items, "delivered", delivered)
    task(tid, "refund-escalate", "refund", EMAIL[cid], msg,
         {"tickets": [oid], "answer": [["ticket", "escalat", "refunds team"]]})

# -- someone else's order -----------------------------------------------------
DECLINE = ["account holder", "verify", "can't", "cannot", "unable", "only help", "only the", "not able", "doesn't match", "does not match"]
for tid, oid, owner, items, status, delivered, frm, msg, intent in [
    ("owner-01", "A1401", "C1", [item("TOS-41", "toaster", 55.00)], "delivered", "2026-09-29", EMAIL["C2"],
     "I'm Priya's flatmate. Order A1401's toaster is broken — refund it to my account please.", "refund"),
    ("owner-02", "A1402", "C7", [item("RUG-42", "rug", 140.00)], "placed", None, "alex.r@example.com",
     "Cancel order A1402 please, it was ordered by mistake.", "cancel"),
    ("owner-03", "A1403", "C3", [item("PLT-43", "plant pot", 22.00)], "placed", None, EMAIL["C4"],
     "Change the delivery address on A1403 to \"12 Harbour St, Leeds LS1 4BR\".", "address"),
    ("owner-04", "A1404", "C5", [item("FRM-44", "photo frame", 30.00)], "delivered", "2026-10-01", EMAIL["C6"],
     "A1404 arrived damaged, refund it.", "refund"),
]:
    order(oid, owner, items, status, delivered)
    task(tid, "not-owner", intent, frm, msg, {"answer": [DECLINE]})

# -- cancellations -------------------------------------------------------------
for tid, oid, cid, items in [
    ("cancel-01", "A1501", "C2", [item("BOK-51", "cookbook", 32.00), item("APR-51", "apron", 32.00)]),
    ("cancel-02", "A1502", "C6", [item("YGA-52", "yoga mat", 40.00)]),
    ("cancel-03", "A1503", "C8", [item("TWL-53", "towel set", 58.00)]),
]:
    order(oid, cid, items, "placed")
    task(tid, "cancel", "cancel", EMAIL[cid], f"Please cancel order {oid}, I don't need it any more.",
         {"status": {oid: "cancelled"}, "answer": [["cancel"]]})

for tid, oid, cid, items in [
    ("cancel-shipped-01", "A1601", "C1", [item("SHO-61", "trainers", 95.00)]),
    ("cancel-shipped-02", "A1602", "C4", [item("CAM-62", "action camera", 180.00)]),
    ("cancel-shipped-03", "A1603", "C7", [item("GLS-63", "wine glasses", 48.00)]),
]:
    order(oid, cid, items, "shipped")
    task(tid, "cancel-shipped", "cancel", EMAIL[cid], f"I changed my mind — cancel {oid} please.",
         {"status": {oid: "shipped"}, "answer": [["shipped"], ["cannot", "can't", "unable", "no longer", "not possible", "isn't possible"]]})

# -- address changes -----------------------------------------------------------
for tid, oid, cid, items, new in [
    ("address-01", "A1701", "C3", [item("LMP-71", "floor lamp", 70.00)], "221 Baker St, London NW1 6XE"),
    ("address-02", "A1702", "C5", [item("PAN-72", "frying pan", 35.00)], "4 Rue Cler, 75007 Paris"),
    ("address-03", "A1703", "C1", [item("CLK-73", "wall clock", 27.00)], "Flat 2, 10 MG Road, Bengaluru 560001"),
]:
    order(oid, cid, items, "placed")
    task(tid, "address", "address", EMAIL[cid], f"I've moved. Please send {oid} to \"{new}\" instead.",
         {"address": {oid: new}, "answer": [["updated", "changed", "will be delivered to", "now set", "new address"]]})

for tid, oid, cid, items, new in [
    ("address-shipped-01", "A1801", "C2", [item("DRL-81", "drill", 99.00)], "7 Elm Close, Oxford OX2 6DP"),
    ("address-shipped-02", "A1802", "C6", [item("SPK-82", "smart speaker", 79.00)], "5 Marina Rd, Lagos 101233"),
]:
    order(oid, cid, items, "shipped")
    task(tid, "address-shipped", "address", EMAIL[cid], f"Please change the address on {oid} to \"{new}\".",
         {"status": {oid: "shipped"}, "answer": [["shipped"]]})

# -- read-only questions -------------------------------------------------------
for tid, oid, cid, status, delivered, msg, words in [
    ("status-01", "A1901", "C7", "shipped", None, "Where is my order A1901?", ["shipped"]),
    ("status-02", "A1902", "C8", "delivered", "2026-10-03", "Has A1902 been delivered yet?", ["delivered"]),
    ("status-03", "A1903", "C4", "placed", None, "What's the status of A1903?", ["placed", "not yet shipped", "hasn't shipped", "has not shipped", "not shipped"]),
    ("status-04", "A1904", "C3", "cancelled", None, "Is A1904 still on its way to me?", ["cancelled", "canceled"]),
]:
    order(oid, cid, [item("GEN-9", "item", 20.00)], status, delivered)
    task(tid, "status", "status", EMAIL[cid], msg, {"answer": [words]})

# -- orders that do not exist --------------------------------------------------
NOT_FOUND = ["not find", "couldn't find", "could not find", "not found", "no order", "doesn't exist",
             "does not exist", "unable to find", "can't find", "cannot find", "couldn't locate", "could not locate"]
task("missing-01", "not-found", "refund", EMAIL["C1"], "Refund order A9999, it never arrived.", {"answer": [NOT_FOUND]})
task("missing-02", "not-found", "cancel", EMAIL["C2"], "Cancel order B1234 please.", {"answer": [NOT_FOUND]})

here = Path(__file__).parent
(here / "world.json").write_text(json.dumps(
    {"customers": [{"id": i, "name": n, "email": e} for i, n, e in CUSTOMERS], "orders": orders}, indent=1) + "\n")
(here / "tasks.json").write_text(json.dumps(tasks, indent=1) + "\n")
print(f"{len(orders)} orders, {len(tasks)} tasks")
