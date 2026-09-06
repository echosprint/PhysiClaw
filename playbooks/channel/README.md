# channel

The conductor's own pack: how to reach YOUR user's thread in the IM app
and speak there. `open` navigates to the thread by in-app search,
`send` pastes and sends a message into it, and `boot/` is the walk every
wake plays first: reach the thread, read the request, hand the matching
playbook the baton.

## Device

Recorded on WeChat with an English system (guards accept both the
English and 中文 labels). The thread page's one anchor is the contact
name in the centered title box, never the top band: an iOS notification
banner prints the same name left-aligned at the same height.

## Traps

- WeChat search can put a "Searched ID" account first; `send` refuses
  to type unless the thread title matches, so a wrong hit aborts.
- `<<CONTACT>>` must be exactly what the app shows as the thread title.
