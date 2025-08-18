Follow `docs\development\windows.md` and then do this to make the login work!

Run this and copy the output to your `.env` file as `HELPER_WIDGET_SECRET`.
```sh
openssl rand -hex 32
```

or

```sh
ruby -rsecurerandom -e 'puts SecureRandom.hex(32)'
```