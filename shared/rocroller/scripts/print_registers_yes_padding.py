import gdb

FUNCTION = "LDS_Address_Modelling_Yes_padding"
THREADS = [99, 100, 101, 102]

PATTERNS = [
    {"offset": 816, "reg": "$v7"},
    {"offset": 904, "reg": "$v9"},
    {"offset": 912, "reg": "$v0"},
    {"offset": 964, "reg": "$v3"},
]

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")

gdb.Breakpoint(FUNCTION, temporary=True)
gdb.execute("run")

for pattern in PATTERNS:
    gdb.Breakpoint(f"*{FUNCTION}+{pattern['offset']}", temporary=True)
    gdb.execute("continue")
    gdb.execute("x/i $pc")
    print(f"# {pattern['reg']}")
    for t in THREADS:
        gdb.execute(f"thread {t}")
        gdb.execute(f"p {pattern['reg']}")

gdb.execute("set confirm off")
gdb.execute("quit")
