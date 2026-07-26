# PhysiClaw

**An AI agent that physically operates a phone — the way you do.**

[English docs](https://docs.physiclaw.ai/en/) · [Demo video](https://youtu.be/YJ1O6gW0KVw)

[中文文档](https://docs.physiclaw.ai/zh/) · [演示视频](https://b23.tv/jnegfgC)

PhysiClaw watches a phone's screen with a camera and taps it with a stylus,
working the phone the way a person would. No APIs, no OAuth, no ADB cables,
nothing installed on the phone — just unlock it, set it on the desk, and let
the agent work.

It's built for the everyday errands that pile up — ordering takeout, shopping
for groceries, booking a ride, paying a bill, replying to a message. Anything
you can do by hand on your phone, PhysiClaw can do for you.

## Why a physical body?

The apps that run your daily life are closed off: most expose no public API,
and simulated input — desktop automation or Android's ADB — leaves software
fingerprints that anti-bot systems flag. So PhysiClaw treats the screen itself
as the API: a camera reads it, a stylus performs the gestures. To the phone
it's indistinguishable from a real finger — nothing to detect — and it reaches
virtually any app reliably.

The trade-off is speed: a few seconds per action, in exchange for universality
and reliability.

## How it works

PhysiClaw has its own dedicated phone running its own WhatsApp account.
Add it as a contact and chat with it like a real person — tell it what you need
in plain language:

```text
Order a latte on DoorDash.
What's the weather tomorrow?
Buy milk and eggs on Instacart.
```

Its runtime loops continuously, waking the agent when a scheduled task is due
or the screen lights up with a new message. The agent unlocks the phone, reads
your message, and does the task — pausing for confirmation when it matters,
like a payment — then replies with the result, saves its memory, and exits
until it's next woken.

## Getting started

You'll need the PhysiClaw hardware — the arm, stylus, and camera — plus an
**iPhone** for it to operate.

The `physiclaw` CLI runs on macOS, Windows, and Linux:

```bash
# Install the CLI (uv + Python 3.12 + physiclaw)
curl -fsSL https://physiclaw.ai/install.sh | bash
# Windows: iwr -useb https://physiclaw.ai/install.ps1 | iex

physiclaw models key <provider>        # add your LLM API key
physiclaw models use <provider/model>  # pick the model the agent runs on
physiclaw doctor                       # check your environment
physiclaw                              # start the server — opens the hardware-setup wizard, then runs the agent
```

## License

[MIT](LICENSE) — both the CAD-as-code hardware and the agent software.
