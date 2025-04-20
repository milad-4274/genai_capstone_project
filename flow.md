## 🛍️ Multi-Agent Travel Planning Process

### 📥 User Input

```
Plan a 7-day adventure trip to Bali for someone traveling from Seville with a budget of $2000.
They enjoy hiking and beaches. The trip should start on April 19th, 2025.
```

---

### 🧠 Context Summarizer (Initial Context)

```yaml
user_preferences*: hiking and beaches
current_location*: Seville
budget*: 2000
destination*: Bali
visa_info: None
itinerary_draft: None
accommodation: None
duration*: 7
start_date*: 2025-04-19
trip_tips: None
destination_activity: None
transportation: None
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `get_location_visa`

- **Agent Input:**
  ```json
  {
    "origin": "Seville",
    "destination": "Bali",
    "other": "None"
  }
  ```

- **Output:**
  ```
  As a Spanish citizen traveling to Bali (Indonesia) for tourism, you don't need a visa for stays up to 30 days.
  Make sure your passport is valid for at least six months upon arrival. A return ticket may be required.
  ```

---

### 🧠 Context Summarizer (After Visa Agent)

```yaml
user_preferences: hiking and beaches
current_location: Seville
budget: 2000
destination: Bali
visa_info*: As a Spanish citizen traveling to Bali ... (no visa required)
itinerary_draft: None
accommodation: None
duration: 7
start_date: 2025-04-19
trip_tips: None
destination_activity: None
transportation: None
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `get_accommodation`

- **Input:**
  ```json
  {
    "destination": "Bali",
    "user preference": "hiking and beaches"
  }
  ```

- **Output:**
  ```
  **Hotel Name:** Munduk Moding Plantation Nature Resort ... [Truncated]
  ```

---

### 🧠 Context Summarizer (After Accommodation Agent)

```yaml
user_preferences: hiking and beaches
current_location: Seville
budget: 2000
destination: Bali
visa_info: As a Spanish citizen traveling to Bali ... (no visa required)
itinerary_draft: None
accommodation*: Munduk Moding Plantation Nature Resort ... [Truncated]
duration: 7
start_date: 2025-04-19
trip_tips: None
destination_activity: None
transportation: None
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `get_transportation`

- **Input:**
  ```json
  {
    "origin": "Seville",
    "destination": "Bali",
    "transportation_preferences": "cheapest and fastest",
    "start_date": "2025-04-19",
    "duration": "7"
  }
  ```

- **Output:**
  ```json
  {
    "transportation": [
      {
        "mode": "Flight",
        "description": "Flights with one or more stops. Look for options with KLM, Qatar Airways or Aegean Airlines... [Truncated]",
        "cost": "Approximately $740 - $1020 USD for round trip."
      }
    ]
  }
  ```

---

### 🧠 Context Summarizer (After Transportation Agent)

```yaml
user_preferences: hiking and beaches
current_location: Seville
budget: 2000
destination: Bali
visa_info: As a Spanish citizen traveling to Bali ... (no visa required)
itinerary_draft: None
accommodation: Munduk Moding Plantation Nature Resort ... [Truncated]
duration: 7
start_date: 2025-04-19
trip_tips: None
destination_activity: None
transportation*:
  - mode: Flight
    description: Flights with one or more stops ... [Truncated]
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `activity_search`

- **Input:**
  ```json
  {
    "search_query": "Bali hiking trails and beaches"
  }
  ```

- **Output:**
  ```
  Based on these search snippets, here are my recommendations ... [Truncated]
  ```

---

### 🧠 Context Summarizer (After Activity Agent)

```yaml
user_preferences: hiking and beaches
current_location: Seville
budget: 2000
destination: Bali
visa_info: As a Spanish citizen traveling to Bali ... (no visa required)
itinerary_draft: None
accommodation: Munduk Moding Plantation Nature Resort ... [Truncated]
duration: 7
start_date: 2025-04-19
trip_tips: None
destination_activity*: Based on these search snippets, here are my recommendations ... [Truncated]
transportation:
  - mode: Flight
    description: Flights with one or more stops ... [Truncated]
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `silly_travel_stylist_structured`

- **Input:**
  ```json
  {
    "destination": "Bali",
    "travel_date": "2025-04-19",
    "duration": "7",
    "preferences": "hiking and beaches",
    "budget": "2000"
  }
  ```

- **Output:**
  ```
  weather_summary, closing_remark, destination, safety_tip, etc [Summarized]
  ```

---

### 🧠 Context Summarizer (After Travel Stylist Agent)

```yaml
user_preferences: hiking and beaches
current_location: Seville
budget: 2000
destination: Bali
visa_info: As a Spanish citizen traveling to Bali ... (no visa required)
itinerary_draft: None
accommodation: Munduk Moding Plantation Nature Resort ... [Truncated]
duration: 7
start_date: 2025-04-19
trip_tips*: weather_summary, closing_remark, destination, safety_tip, etc [Summarized]
destination_activity: Based on these search snippets, here are my recommendations ... [Truncated]
transportation:
  - mode: Flight
    description: Flights with one or more stops ... [Truncated]
```

---

### 🧑‍💼 Supervisor

#### 🔹 Agent: `generate_itinerary`

- **Input:**
  All accumulated context as JSON
- **Output:**
  ```
  Structured itinerary
  ```

---


