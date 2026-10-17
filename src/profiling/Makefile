CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

TARGET = profile_harness
SRC = profile_harness.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
