---
applies_to: [java]
---

## Java Test Generation Guide (JUnit 5)

Write JUnit 5 (`jupiter`) tests for the Java source files just modified. **Do not use Mockito or any other mocking framework.** Call the real implementations; use the real types. When a side effect is genuinely impractical to invoke, prefer an in-process real resource (an H2 in-memory DB, a `@TempDir` directory, `WireMock` only as a real local HTTP server — not as a mock).

### File placement
- Maven layout: `src/main/java/com/x/Y.java` → `src/test/java/com/x/YTest.java`. Same package.
- Gradle defaults to the same layout.

### Structure
- One test class per source class: `class YTest`.
- Test methods: `@Test void <behaviour_under_test>()` — descriptive names, no `should` prefix.
- `@BeforeEach` / `@AfterEach` for per-test setup. `@BeforeAll` / `@AfterAll` (static) for once-per-class.
- `@ParameterizedTest` + `@CsvSource` / `@MethodSource` when the same logic is exercised across inputs.

### Real fakes from JUnit / stdlib
- `@TempDir Path tempDir` — JUnit provides a scoped temp directory.
- `System.setProperty` / `System.clearProperty` — wrap in try/finally or `@AfterEach`.
- H2 in-memory DB (`jdbc:h2:mem:test;DB_CLOSE_DELAY=-1`) for repository tests.

### Assertions
- `org.junit.jupiter.api.Assertions.*`:
  - `assertEquals(expected, actual)` / `assertNotEquals`.
  - `assertThrows(IllegalArgumentException.class, () -> y.doIt(bad))`.
  - `assertTrue` / `assertFalse` — prefer the specific form where possible.
- `assertAll(...)` to bundle related assertions so all fire even when one fails.

### Dates and times — a test must not depend on the day it runs
A test whose expected value changes with the calendar passes today and fails
months later with no code change. `Clock.fixed` is a real stdlib implementation,
not a mock, so the no-mocking rule does not forbid it — this is exactly the
"real fakes from JUnit / stdlib" rule above.

```java
Clock fixed = Clock.fixed(Instant.parse("2026-03-01T00:00:00Z"), ZoneOffset.UTC);
assertEquals(2027, service.nextOccurrence(MonthDay.of(2, 10), fixed).getYear());
```

If the production code reads `LocalDate.now()` with no injectable `Clock`, that
is an actual design problem and taking a `Clock` parameter is the correct fix —
unlike adding a parameter solely to inject a fake.

Or assert the invariant rather than a literal arrangement:
```java
assertEquals(dates.stream().sorted().toList(), dates);   // "sorted by date"
```

Never assert a hardcoded order or value that depends on today — recurring
dates, ages, "upcoming"/"within N days" windows, expiry, relative buckets
("this week"), and timezone-dependent day boundaries all qualify.

### What NOT to do
- No `@Mock`, no `Mockito.mock(...)`, no `Mockito.when(...).thenReturn(...)`, no `@InjectMocks`.
- No assertion whose expected value depends on today's date — see "Dates and times" above.
- No PowerMock, no EasyMock, no JMockit.
- Do not introduce a constructor parameter solely to inject a fake — refactor the production code only if it has an actual design problem, not for the test.

### Minimal example
```java
package com.x;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {

    @Test
    void returnsQuotientForIntegers() {
        assertEquals(5, Calculator.divide(10, 2));
    }

    @Test
    void throwsOnZeroDivisor() {
        ArithmeticException ex = assertThrows(
            ArithmeticException.class,
            () -> Calculator.divide(1, 0)
        );
        assertTrue(ex.getMessage().contains("cannot divide by zero"));
    }

    @ParameterizedTest
    @CsvSource({"0,1,0", "-4,2,-2", "7,7,1"})
    void table(int a, int b, int expected) {
        assertEquals(expected, Calculator.divide(a, b));
    }
}
```
