"""Integration tests for MedVision channel + HiperHealth pipeline.

Tests verify that SkinAnalysisSkill works end-to-end through
HiperHealth's StageRunner, PipelineContext, Session, and
SkillRegistry APIs — not just in isolation.
"""

from __future__ import annotations

from pathlib import Path

from hiperhealth.pipeline import (
    BaseSkill,
    PipelineContext,
    SkillMetadata,
    Stage,
    StageRunner,
)
from hiperhealth.pipeline.session import Session

from skills.skin_analysis.skill import SkinAnalysisSkill
from tests.conftest import _mock_skin_patches

# ── Lightweight stub skills for multi-skill tests ──────


class _StubPrivacySkill(BaseSkill):
    """Minimal stand-in for the built-in PrivacySkill.

    Needed because the real PrivacySkill may not be importable
    in the medvision-channel test environment.  Matches the same
    stages as the real one (screening, intake).
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='hiperhealth.privacy',
                stages=('screening', 'intake'),
            )
        )

    def pre(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        return ctx

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.extras.setdefault('privacy', {})['ran'] = True
        return ctx


class _StubDiagnosticsSkill(BaseSkill):
    """Minimal stand-in for DiagnosticsSkill.

    Reads ``ctx.extras['prompt_fragments']['diagnosis']``
    to verify that MedVision's injection actually works.
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='hiperhealth.diagnostics',
                stages=('diagnosis', 'exam'),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        if stage != 'diagnosis':
            return ctx

        extra = ctx.extras.get('prompt_fragments', {}).get('diagnosis', '')
        ctx.results.setdefault(stage, {})['llm_received_visual'] = bool(extra)
        return ctx


# ── Test Class 1: StageRunner Integration ──────────────


class TestStageRunnerIntegration:
    """Pipeline hook execution with SkinAnalysisSkill."""

    def test_runner_calls_intake_hooks(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        pipeline_ctx_with_image: PipelineContext,
    ) -> None:
        """Running intake produces audit entries for all three hooks."""
        runner = StageRunner(skills=[mock_skin_skill])

        result = runner.run(
            Stage.INTAKE,
            pipeline_ctx_with_image,
        )

        skill_audits = [
            a
            for a in result.audit
            if a.skill_name == 'medvision.skin_analysis'
        ]
        hooks = [a.hook for a in skill_audits]
        assert hooks == ['pre', 'execute', 'post']

    def test_runner_skips_irrelevant_stages(
        self,
        mock_skin_skill: SkinAnalysisSkill,
    ) -> None:
        """SkinAnalysisSkill should NOT run during screening."""
        runner = StageRunner(skills=[mock_skin_skill])
        ctx = PipelineContext(patient={})

        result = runner.run(Stage.SCREENING, ctx)

        skin_audits = [
            a
            for a in result.audit
            if a.skill_name == 'medvision.skin_analysis'
        ]
        assert len(skin_audits) == 0

    def test_registration_order_with_stubs(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        pipeline_ctx_with_image: PipelineContext,
    ) -> None:
        """Skills execute in registration order, grouped by hook.

        For intake stage with privacy + skin_analysis:
        all pre hooks → all execute hooks → all post hooks,
        each group in registration order.
        """
        privacy = _StubPrivacySkill()
        runner = StageRunner(
            skills=[privacy, mock_skin_skill],
        )

        result = runner.run(
            Stage.INTAKE,
            pipeline_ctx_with_image,
        )

        pre_audits = [a.skill_name for a in result.audit if a.hook == 'pre']
        exec_audits = [
            a.skill_name for a in result.audit if a.hook == 'execute'
        ]
        post_audits = [a.skill_name for a in result.audit if a.hook == 'post']

        assert pre_audits == [
            'hiperhealth.privacy',
            'medvision.skin_analysis',
        ]
        assert exec_audits == [
            'hiperhealth.privacy',
            'medvision.skin_analysis',
        ]
        assert post_audits == [
            'hiperhealth.privacy',
            'medvision.skin_analysis',
        ]


# ── Test Class 2: Prompt Fragment Injection ────────────


class TestPromptFragmentInjection:
    """The core integration: intake → diagnosis data flow."""

    def test_intake_then_diagnosis_injects_fragments(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        pipeline_ctx_with_image: PipelineContext,
    ) -> None:
        """Two-stage handoff: execute(intake) → pre(diagnosis).

        execute("intake") writes visual_observations, then
        pre("diagnosis") reads them and writes prompt_fragments.
        """
        diagnostics = _StubDiagnosticsSkill()
        runner = StageRunner(
            skills=[mock_skin_skill, diagnostics],
        )

        ctx = runner.run(
            Stage.INTAKE,
            pipeline_ctx_with_image,
        )
        assert 'visual_observations' in ctx.results['intake']

        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert 'prompt_fragments' in ctx.extras
        fragment = ctx.extras['prompt_fragments']['diagnosis']
        assert 'Visual Observations' in fragment
        assert ctx.results['diagnosis']['llm_received_visual']

    def test_run_many_intake_diagnosis(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        pipeline_ctx_with_image: PipelineContext,
    ) -> None:
        """run_many sequentially runs intake then diagnosis."""
        diagnostics = _StubDiagnosticsSkill()
        runner = StageRunner(
            skills=[mock_skin_skill, diagnostics],
        )

        ctx = runner.run_many(
            [Stage.INTAKE, Stage.DIAGNOSIS],
            pipeline_ctx_with_image,
        )

        assert 'intake' in ctx.results
        assert 'diagnosis' in ctx.results
        assert 'prompt_fragments' in ctx.extras

    def test_disabled_skill_skips_visual(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        pipeline_ctx_with_image: PipelineContext,
    ) -> None:
        """A/B testing: disabled skill produces no observations."""
        diagnostics = _StubDiagnosticsSkill()
        runner = StageRunner(
            skills=[mock_skin_skill, diagnostics],
        )

        with runner.disabled({'medvision.skin_analysis'}):
            ctx = runner.run_many(
                [Stage.INTAKE, Stage.DIAGNOSIS],
                pipeline_ctx_with_image,
            )

        assert 'visual_observations' not in ctx.results.get('intake', {})
        assert 'prompt_fragments' not in ctx.extras


# ── Test Class 3: Check Requirements ──────────────────


class TestCheckRequirements:
    """Inquiry flow through StageRunner + Session."""

    def test_check_requirements_returns_image_inquiry(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        tmp_session: Session,
    ) -> None:
        """Session without skin_image triggers image inquiry."""
        runner = StageRunner(skills=[mock_skin_skill])

        inquiries = runner.check_requirements(
            Stage.INTAKE,
            tmp_session,
        )

        image_inqs = [i for i in inquiries if i.field == 'skin_image']
        assert len(image_inqs) == 1
        assert image_inqs[0].input_type == 'image'
        assert image_inqs[0].priority == 'required'

    def test_check_requirements_satisfied_after_answer(
        self,
        mock_skin_skill: SkinAnalysisSkill,
        tmp_session: Session,
    ) -> None:
        """Providing skin_image clears required inquiries."""
        runner = StageRunner(skills=[mock_skin_skill])

        tmp_session.provide_answers(
            {'skin_image': '/path/to/img.jpg'},
        )

        inquiries = runner.check_requirements(
            Stage.INTAKE,
            tmp_session,
        )
        required = [i for i in inquiries if i.priority == 'required']
        assert len(required) == 0


# ── Test Class 4: Session Integration ─────────────────


class TestSessionIntegration:
    """Parquet session persistence of visual observations."""

    def test_session_persists_visual_observations(
        self,
        tmp_path: Path,
    ) -> None:
        """Visual observations survive Parquet round-trip.

        Creates a session, runs intake (mocked), then loads
        a fresh Session from the same file and checks that
        results are preserved.
        """
        session_path = tmp_path / 'persist_test.parquet'
        session = Session.create(str(session_path))
        session.set_clinical_data(
            {
                'skin_image': '/fake/img.jpg',
                'age': 45,
            }
        )

        with _mock_skin_patches():
            skill = SkinAnalysisSkill()
            runner = StageRunner(skills=[skill])
            runner.run_session('intake', session)

        reloaded = Session.load(str(session_path))
        results = reloaded.results

        assert 'intake' in results
        intake = results['intake']
        assert 'visual_observations' in intake
        obs = intake['visual_observations'][0]
        assert obs['skill'] == 'medvision.skin_analysis'

    def test_session_records_stage_events(
        self,
        tmp_path: Path,
    ) -> None:
        """Session event log records stage_started + stage_completed."""
        session_path = tmp_path / 'events_test.parquet'
        session = Session.create(str(session_path))
        session.set_clinical_data(
            {
                'skin_image': '/fake/img.jpg',
                'age': 45,
            }
        )

        with _mock_skin_patches():
            skill = SkinAnalysisSkill()
            runner = StageRunner(skills=[skill])
            runner.run_session('intake', session)

        reloaded = Session.load(str(session_path))
        event_types = [e['event_type'] for e in reloaded.events]

        assert 'stage_started' in event_types
        assert 'stage_completed' in event_types
        assert 'intake' in reloaded.stages_completed


# ── Test Class 5: Context Serialization ───────────────


class TestContextSerialization:
    """JSON round-trip of visual observation data."""

    def test_visual_observation_survives_json(self) -> None:
        """Nested observation dicts round-trip through JSON."""
        ctx = PipelineContext(
            patient={'skin_image': '/path/img.jpg'},
            session_id='ser-test',
        )
        ctx.results['intake'] = {
            'visual_observations': [
                {
                    'skill': 'medvision.skin_analysis',
                    'finding': 'melanocytic_nevi',
                    'display_name': 'Melanocytic Nevi',
                    'confidence': 0.92,
                    'calibrated': True,
                    'body_site': 'skin',
                    'snomed_code': '400010006',
                    'heatmap_path': None,
                    'status': 'preliminary',
                    'requires_review': False,
                    'class_probabilities': {
                        'melanocytic_nevi': 0.92,
                        'melanoma': 0.04,
                        'benign_keratosis': 0.02,
                    },
                }
            ]
        }

        json_str = ctx.model_dump_json()
        restored = PipelineContext.model_validate_json(json_str)

        obs_orig = ctx.results['intake']['visual_observations'][0]
        obs_rest = restored.results['intake']['visual_observations'][0]
        assert obs_rest['finding'] == obs_orig['finding']
        assert obs_rest['confidence'] == obs_orig['confidence']
        assert obs_rest['heatmap_path'] is None
        assert (
            obs_rest['class_probabilities'] == obs_orig['class_probabilities']
        )

    def test_prompt_fragments_survive_json(self) -> None:
        """prompt_fragments text round-trips through JSON."""
        ctx = PipelineContext(patient={})
        ctx.extras['prompt_fragments'] = {
            'diagnosis': (
                '## Visual Observations\n'
                '- **Primary finding:** Melanoma\n'
                '  - Confidence: 85% (calibrated)\n'
            )
        }

        json_str = ctx.model_dump_json()
        restored = PipelineContext.model_validate_json(json_str)

        assert (
            restored.extras['prompt_fragments']['diagnosis']
            == ctx.extras['prompt_fragments']['diagnosis']
        )
