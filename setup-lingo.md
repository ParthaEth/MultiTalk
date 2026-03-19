


```sh
celery -A lingo_video_worker:app worker -l info -P solo -n multitalk@%h
```
