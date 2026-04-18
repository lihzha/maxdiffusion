First request a TPU with:

```bash
export TPU_NAME=you_tpu_name
tpu watch v6 -n 8
```

After verifying that the tpu is already successfully requested with tpu list:

```bash
tpu watch v6 -f -n 8 \
  --setup-cmd ...

```

For single host you can ssh into the VM


```bash
tpu attach v6
```

If command failed, run

```bash
tpu nuke v6
```