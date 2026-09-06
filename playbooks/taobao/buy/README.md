# taobao/buy

From the user's message to a paid order: `parse` derives the search
keyword, `launch` cold-starts Taobao, `search` types the keyword with
非直播 appended, the `pick` agent walks results to the buy sheet and
stops before payment, `confirm-pay` quotes the sheet's exact total and
waits for 好的 / 不用, `pay` is one recorded tap, `report` tells the user.

## Recorded facts

- Search bar, paste bubble, and 搜索 key coordinates come from the
  results screen of the recording device; the paste bubble heals by
  its text.
- The pay button (免密支付 / 立即支付 / 提交订单) sits at the sheet's
  bottom; a two-step cashier gets its second tap only when 立即支付 or
  确认支付 shows.
- The pick prompt asks for a single pack matching the keyword and
  escalates when the matching spec is 缺货 or the sheet cannot show one
  exact price.

## Rehearsal

    physiclaw playbooks replay taobao/buy --session <id> -i user_said=... -o parse.keyword=...
    physiclaw playbooks run taobao/buy -i 'user_said=帮我在淘宝买一袋五常大米，5kg装的'
