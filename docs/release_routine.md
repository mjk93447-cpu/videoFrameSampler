# Release Routine Template

## Purpose

Reusable release checklist for each version (`ver1`, `ver2`, ...).

## Steps

1. **Run validation**
   - `python -m pytest -q`
   - Manual smoke test of GUI extraction
2. **Update version/docs**
   - Reflect version label in `README.md` and relevant docs
   - Update change log notes
3. **Commit and push**
   - `git add .`
   - `git commit -m "release: finalize verX build and docs"`
   - `git push`
4. **Trigger/verify GitHub Actions**
   - Confirm `.github/workflows/build-exe.yml` run success
   - Download and smoke-test EXE artifact
5. **Publish release note**
   - Include test summary, known limitations, artifact link

## Verification Record (per release)

- Version label:
- Commit SHA:
- Test result:
- Action run URL:
- Artifact validation note:
