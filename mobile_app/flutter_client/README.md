# SENSE Mobile App (Flutter)

**Role:** Student 3 — Mobile / Flutter Engineer
**Stack:** Flutter 3.16+ · Dart · Provider

The mobile app is the **UI + audio client only**. All AI processing
happens on the backend. The app records audio, sends it to the server,
and displays Teta AI's responses.

## Architecture

```
┌─────────────────────────────────┐
│         SENSE Mobile App        │
│                                 │
│  WelcomeScreen → ChatScreen     │
│       │              │          │
│  ChatProvider (state management)│
│       │              │          │
│  ApiService      AudioService   │
│       │              │          │
│   REST API       Microphone     │
│       │          Recording      │
└───────┼──────────────┘          │
        │                         │
        ▼                         │
   Backend Server (Student 2)     │
```

**RULE:** The app NEVER calls OpenAI, Whisper, or any AI model directly.
Everything goes through: `Mobile → Backend → AI Pipeline → Backend → Mobile`

## Folder structure

```
lib/
├── main.dart                    # App entry point
├── models/
│   ├── chat_message.dart        # Conversation turn model
│   └── session.dart             # Session + analysis models
├── screens/
│   ├── welcome_screen.dart      # Landing / start screen
│   └── chat_screen.dart         # Main conversation UI
├── services/
│   ├── api_service.dart         # REST API communication
│   ├── audio_service.dart       # Microphone + playback
│   └── chat_provider.dart       # State management (Provider)
├── theme/
│   └── sense_theme.dart         # Colors, fonts, dimensions
└── widgets/
    ├── chat_bubble.dart         # Message bubbles
    ├── voice_button.dart        # Hold-to-talk button
    └── connection_status.dart   # Server status indicator
```

## Setup

### Prerequisites
- Flutter SDK 3.16+
- Android Studio or Xcode
- The SENSE backend running (see backend/README.md)

### Run

```bash
cd mobile_app/flutter_client

# Install dependencies
flutter pub get

# Run on connected device / emulator
flutter run
```

### Backend connection
The app auto-detects the platform:
- **Android emulator** → connects to `http://10.0.2.2:8000`
- **iOS simulator** → connects to `http://127.0.0.1:8000`
- **Real device** → change the IP in `lib/services/api_service.dart` to your computer's LAN IP

### Platform permissions
Before building, add the required permissions:

**Android** (`android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```
Add `android:usesCleartextTraffic="true"` to `<application>` for local dev HTTP.

**iOS** (`ios/Runner/Info.plist`):
```xml
<key>NSMicrophoneUsageDescription</key>
<string>SENSE needs microphone access so Teta can hear you speak.</string>
```

## Conversation flow

1. **Welcome screen** — checks server connection, child taps "يلّا نحكي مع تيتا"
2. **Chat screen** — child holds the mic button to talk, releases to send
3. **Backend processes** — audio → Whisper → Teta AI → reply text
4. **Reply appears** — as a chat bubble from Teta
5. **End session** — final analysis runs, risk level shown

## Design principles

- **Child-friendly**: rounded shapes, pastel colors, large buttons
- **RTL-first**: Arabic right-to-left layout throughout
- **Safe**: no clinical language shown to the child
- **Minimal**: one button to talk, one to end — no complexity
