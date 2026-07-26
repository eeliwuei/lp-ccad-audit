# Release and tag policy

Published Git tags are records, not moving aliases.

* Never force-move or delete a tag after it has been cited in a manuscript,
  archive or release manifest.
* A corrected submission receives a new versioned tag, such as
  `v1.0.1-submission`; it does not reuse `v1.0-submission`.
* A release manifest records both the tag object ID (for an annotated tag) and
  the peeled commit ID. A tag name by itself is not sufficient evidence because
  Git hosting services permit privileged users to move it.
* Generate a manifest only after the release commit and annotated tag exist.
  Store the generated manifest outside this repository so the manifest does not
  change the commit it describes.

From a clean clone, generate the external manifest with:

```bash
python3 scripts/generate_release_manifest.py \
  --ref v1.0.1-submission \
  > ../PUBLIC_RELEASE_MANIFEST.md
```

Verify the recorded object IDs from another clone before citing the release.
