"""Tests for load_public_skills functionality with git-based caching."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from openhands.sdk.context.agent_context import AgentContext
from openhands.sdk.context.skills import (
    KeywordTrigger,
    Skill,
    load_public_skills,
)
from openhands.sdk.context.skills.utils import update_skills_repository


@pytest.fixture
def mock_repo_dir(tmp_path):
    """Create a mock git repository with skills and marketplace."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Create skills directory
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create skill files
    git_skill = skills_dir / "git.md"
    git_skill.write_text(
        "---\n"
        "name: git\n"
        "triggers:\n"
        "  - git\n"
        "  - github\n"
        "---\n"
        "Git best practices and commands."
    )

    docker_skill = skills_dir / "docker.md"
    docker_skill.write_text(
        "---\n"
        "name: docker\n"
        "triggers:\n"
        "  - docker\n"
        "  - container\n"
        "---\n"
        "Docker guidelines and commands."
    )

    testing_skill = skills_dir / "testing.md"
    testing_skill.write_text(
        "---\nname: testing\n---\nTesting guidelines for all repos."
    )

    # Create marketplace
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()
    marketplace = {
        "name": "default",
        "owner": {"name": "Test", "email": "test@test.com"},
        "plugins": [
            {"name": "git", "source": "./git"},
            {"name": "docker", "source": "./docker"},
            {"name": "testing", "source": "./testing"},
        ],
    }
    (marketplaces_dir / "default.json").write_text(json.dumps(marketplace))

    # Create .git directory to simulate a git repo
    git_dir = repo_dir / ".git"
    git_dir.mkdir()

    return repo_dir


@pytest.fixture
def mock_repo_with_agentskills_references(tmp_path):
    """Create a mock repo with AgentSkills-style skills with reference markdown files.

    This reproduces the issue where markdown files in subdirectories of a SKILL.md
    directory (like themes/ or references/) are incorrectly loaded as separate skills.
    See: https://github.com/OpenHands/software-agent-sdk/issues/1981
    """
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Create skills directory
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create theme-factory skill with SKILL.md and reference markdown files in themes/
    theme_factory_dir = skills_dir / "theme-factory"
    theme_factory_dir.mkdir()

    # Main SKILL.md file
    skill_md = theme_factory_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: theme-factory\n"
        "description: Toolkit for styling artifacts with a theme.\n"
        "---\n"
        "# Theme Factory Skill\n\n"
        "This skill provides a curated collection of professional themes.\n"
    )

    # Create themes subdirectory with reference markdown files
    themes_dir = theme_factory_dir / "themes"
    themes_dir.mkdir()

    # These are reference files, NOT separate skills
    (themes_dir / "arctic-frost.md").write_text(
        "# Arctic Frost\n\nA cool and crisp winter-inspired theme.\n"
    )
    (themes_dir / "ocean-depths.md").write_text(
        "# Ocean Depths\n\nA professional and calming maritime theme.\n"
    )
    (themes_dir / "sunset-boulevard.md").write_text(
        "# Sunset Boulevard\n\nWarm and vibrant sunset colors.\n"
    )

    # Create readiness-report skill with references/ subdirectory
    readiness_dir = skills_dir / "readiness-report"
    readiness_dir.mkdir()

    (readiness_dir / "SKILL.md").write_text(
        "---\n"
        "name: readiness-report\n"
        "description: Generate readiness reports.\n"
        "---\n"
        "# Readiness Report Skill\n"
    )

    # Create references subdirectory with reference markdown files
    refs_dir = readiness_dir / "references"
    refs_dir.mkdir()

    (refs_dir / "criteria.md").write_text("# Criteria\n\nEvaluation criteria.\n")
    (refs_dir / "maturity-levels.md").write_text(
        "# Maturity Levels\n\nMaturity level definitions.\n"
    )

    # Create a regular legacy skill (not AgentSkills format)
    legacy_skill = skills_dir / "legacy-skill.md"
    legacy_skill.write_text(
        "---\nname: legacy-skill\ntriggers:\n  - legacy\n---\nA legacy format skill.\n"
    )

    # Create marketplace
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()
    marketplace = {
        "name": "default",
        "owner": {"name": "Test", "email": "test@test.com"},
        "plugins": [
            {"name": "theme-factory", "source": "./theme-factory"},
            {"name": "readiness-report", "source": "./readiness-report"},
            {"name": "legacy-skill", "source": "./legacy-skill"},
        ],
    }
    (marketplaces_dir / "default.json").write_text(json.dumps(marketplace))

    # Create .git directory to simulate a git repo
    git_dir = repo_dir / ".git"
    git_dir.mkdir()

    return repo_dir


def test_load_public_skills_success(mock_repo_dir):
    """Test successfully loading skills from local marketplace file."""
    marketplace_file = mock_repo_dir / "marketplaces" / "default.json"
    skills = load_public_skills(str(marketplace_file))

    assert len(skills) == 3
    skill_names = {s.name for s in skills}
    assert skill_names == {"git", "docker", "testing"}

    # Check git skill details
    git_skill = next(s for s in skills if s.name == "git")
    assert isinstance(git_skill.trigger, KeywordTrigger)
    assert "git" in git_skill.trigger.keywords

    # Check testing skill (no trigger - always active)
    testing_skill = next(s for s in skills if s.name == "testing")
    assert testing_skill.trigger is None


def test_load_public_skills_repo_update_fails(tmp_path):
    """Test handling when repository update fails."""

    def mock_update_repo(repo_url, branch, cache_dir):
        return None

    with (
        patch(
            "openhands.sdk.context.skills.skill.update_skills_repository",
            side_effect=mock_update_repo,
        ),
        patch(
            "openhands.sdk.context.skills.skill.get_skills_cache_dir",
            return_value=tmp_path,
        ),
    ):
        skills = load_public_skills()
        assert skills == []


def test_load_public_skills_no_skills_directory(tmp_path):
    """Test handling when skills directory doesn't exist in repo."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    # No skills directory created

    def mock_update_repo(repo_url, branch, cache_dir):
        return repo_dir

    with (
        patch(
            "openhands.sdk.context.skills.skill.update_skills_repository",
            side_effect=mock_update_repo,
        ),
        patch(
            "openhands.sdk.context.skills.skill.get_skills_cache_dir",
            return_value=tmp_path,
        ),
    ):
        skills = load_public_skills()
        assert skills == []


def test_load_public_skills_with_invalid_skill(tmp_path):
    """Test that invalid skills are skipped gracefully."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Valid skill
    valid_dir = skills_dir / "valid"
    valid_dir.mkdir()
    (valid_dir / "SKILL.md").write_text("---\nname: valid\n---\nValid skill content.")

    # Invalid skill
    invalid_dir = skills_dir / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "SKILL.md").write_text(
        "---\nname: invalid\ntriggers: not_a_list\n---\nInvalid skill."
    )

    # Create marketplace that references both skills
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()
    marketplace = {
        "name": "default",
        "owner": {"name": "Test", "email": "test@test.com"},
        "plugins": [
            {"name": "valid", "source": "./valid"},
            {"name": "invalid", "source": "./invalid"},
        ],
    }
    marketplace_file = marketplaces_dir / "default.json"
    marketplace_file.write_text(json.dumps(marketplace))

    skills = load_public_skills(str(marketplace_file))
    # Only valid skill should be loaded, invalid one skipped
    assert len(skills) == 1
    assert skills[0].name == "valid"


def test_update_skills_repository_clone_new(tmp_path):
    """Test cloning a new repository."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch(
        "openhands.sdk.git.utils.subprocess.run", return_value=mock_result
    ) as mock_run:
        repo_path = update_skills_repository(
            "https://github.com/OpenHands/extensions",
            "main",
            cache_dir,
        )

        assert repo_path is not None
        # Check that git clone was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "git"
        assert call_args[0][0][1] == "clone"
        assert "--branch" in call_args[0][0]
        assert "main" in call_args[0][0]


def test_update_skills_repository_update_existing(tmp_path):
    """Test updating an existing repository."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create existing repo with .git directory
    # Cache name derived from URL: OpenHands/extensions -> openhands-extensions
    repo_path = cache_dir / "openhands-extensions"
    repo_path.mkdir()
    git_dir = repo_path / ".git"
    git_dir.mkdir()

    mock_result = MagicMock()
    mock_result.returncode = 0
    # Simulate being on a branch (not detached HEAD) so reset is called
    mock_result.stdout = "main"

    with patch(
        "openhands.sdk.git.utils.subprocess.run", return_value=mock_result
    ) as mock_run:
        result_path = update_skills_repository(
            "https://github.com/OpenHands/extensions",
            "main",
            cache_dir,
        )

        assert result_path == repo_path
        # The git operations are: fetch, checkout, get_current_branch, reset
        # (get_current_branch returns branch name so reset is called)
        assert mock_run.call_count == 4
        all_commands = [call[0][0] for call in mock_run.call_args_list]
        assert all_commands[0][:3] == ["git", "fetch", "origin"]
        assert all_commands[1][:2] == ["git", "checkout"]
        assert all_commands[2] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        assert all_commands[3][:3] == ["git", "reset", "--hard"]


def test_update_skills_repository_clone_timeout(tmp_path):
    """Test handling of timeout during clone."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch(
        "openhands.sdk.git.utils.subprocess.run",
        side_effect=subprocess.TimeoutExpired("git", 60),
    ) as mock_run:
        repo_path = update_skills_repository(
            "https://github.com/OpenHands/extensions",
            "main",
            cache_dir,
        )

        assert repo_path is None
        mock_run.assert_called_once()


def test_update_skills_repository_update_fails_uses_cache(tmp_path):
    """Test that existing cache is used when update fails."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create existing repo with .git directory
    # Cache name derived from URL: OpenHands/extensions -> openhands-extensions
    repo_path = cache_dir / "openhands-extensions"
    repo_path.mkdir()
    git_dir = repo_path / ".git"
    git_dir.mkdir()

    # Mock subprocess.run to return a failed result (non-zero return code)
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: fetch failed"

    with patch(
        "openhands.sdk.git.utils.subprocess.run",
        return_value=mock_result,
    ):
        result_path = update_skills_repository(
            "https://github.com/OpenHands/extensions",
            "main",
            cache_dir,
        )

        # Should still return the cached path even though update failed
        assert result_path == repo_path


def test_agent_context_loads_public_skills(mock_repo_dir):
    """Test that AgentContext loads public skills when enabled."""
    marketplace_file = mock_repo_dir / "marketplaces" / "default.json"
    context = AgentContext(load_public_skills=True, marketplace=str(marketplace_file))
    skill_names = {s.name for s in context.skills}
    assert "git" in skill_names
    assert "docker" in skill_names
    assert "testing" in skill_names


def test_agent_context_can_disable_public_skills_loading():
    """Test that public skills loading can be disabled."""
    context = AgentContext(load_public_skills=False)
    assert context.skills == []


def test_agent_context_merges_explicit_and_public_skills(mock_repo_dir):
    """Test that explicit skills and public skills are merged correctly."""
    marketplace_file = mock_repo_dir / "marketplaces" / "default.json"

    # Create explicit skill
    explicit_skill = Skill(
        name="explicit_skill",
        content="Explicit skill content.",
        trigger=None,
    )

    context = AgentContext(
        skills=[explicit_skill],
        load_public_skills=True,
        marketplace=str(marketplace_file),
    )
    skill_names = {s.name for s in context.skills}
    assert "explicit_skill" in skill_names
    assert "git" in skill_names
    assert len(context.skills) == 4  # 1 explicit + 3 public


def test_agent_context_explicit_skill_takes_precedence(mock_repo_dir):
    """Test that explicitly provided skills take precedence over public skills."""
    marketplace_file = mock_repo_dir / "marketplaces" / "default.json"

    # Create explicit skill with same name as public skill
    explicit_skill = Skill(
        name="git",
        content="Explicit git skill content.",
        trigger=None,
    )

    context = AgentContext(
        skills=[explicit_skill],
        load_public_skills=True,
        marketplace=str(marketplace_file),
    )
    # Should have 3 skills (1 explicit git + 2 other public skills)
    assert len(context.skills) == 3
    git_skill = next(s for s in context.skills if s.name == "git")
    # Explicit skill should be used, not the public skill
    assert git_skill.content == "Explicit git skill content."


def test_load_public_skills_excludes_reference_markdown_in_agentskills_folders(
    mock_repo_with_agentskills_references,
):
    """Test that markdown files in SKILL.md subdirs are NOT loaded as skills.

    This is a regression test for issue #1981:
    https://github.com/OpenHands/software-agent-sdk/issues/1981

    When a skill directory contains a SKILL.md file (AgentSkills format), any
    markdown files in subdirectories (like themes/, references/, etc.) should
    be treated as reference materials for that skill, NOT as separate skills.

    Expected behavior:
    - theme-factory/SKILL.md -> loaded as "theme-factory" skill
    - theme-factory/themes/*.md -> NOT loaded (reference files)
    - readiness-report/SKILL.md -> loaded as "readiness-report" skill
    - readiness-report/references/*.md -> NOT loaded (reference files)
    - legacy-skill.md -> loaded as "legacy-skill" skill
    """
    marketplace_file = (
        mock_repo_with_agentskills_references / "marketplaces" / "default.json"
    )
    skills = load_public_skills(str(marketplace_file))

    # Get all skill names
    skill_names = {s.name for s in skills}

    # Should have exactly 3 skills: theme-factory, readiness-report, legacy-skill
    assert len(skills) == 3, (
        f"Expected 3 skills but got {len(skills)}. "
        f"Skill names: {skill_names}. "
        "Reference markdown files in themes/ or references/ subdirectories "
        "should NOT be loaded as separate skills."
    )

    # Verify the correct skills are loaded
    assert "theme-factory" in skill_names
    assert "readiness-report" in skill_names
    assert "legacy-skill" in skill_names

    # Verify reference files are NOT loaded as skills
    # These would be loaded with names like "theme-factory/themes/arctic-frost"
    for skill in skills:
        assert "arctic-frost" not in skill.name, (
            f"Reference arctic-frost.md loaded as skill: {skill.name}"
        )
        assert "ocean-depths" not in skill.name, (
            f"Reference ocean-depths.md loaded as skill: {skill.name}"
        )
        assert "sunset-boulevard" not in skill.name, (
            f"Reference sunset-boulevard.md loaded as skill: {skill.name}"
        )
        assert "criteria" not in skill.name, (
            f"Reference criteria.md loaded as skill: {skill.name}"
        )
        assert "maturity-levels" not in skill.name, (
            f"Reference maturity-levels.md loaded as skill: {skill.name}"
        )


# Tests for marketplace-based skill filtering


@pytest.fixture
def mock_repo_with_marketplace(tmp_path):
    """Create a mock git repository with marketplace file and skills."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Create skills directory
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create marketplaces directory
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()

    # Create multiple skills (some in marketplace, some not)
    # Skill 1: git (in marketplace)
    git_dir = skills_dir / "git"
    git_dir.mkdir()
    (git_dir / "SKILL.md").write_text(
        "---\nname: git\ndescription: Git best practices\n---\nGit skill content."
    )

    # Skill 2: docker (in marketplace)
    docker_dir = skills_dir / "docker"
    docker_dir.mkdir()
    (docker_dir / "SKILL.md").write_text(
        "---\nname: docker\ndescription: Docker guidelines\n---\nDocker skill content."
    )

    # Skill 3: internal-only (NOT in marketplace)
    internal_dir = skills_dir / "internal-only"
    internal_dir.mkdir()
    (internal_dir / "SKILL.md").write_text(
        "---\nname: internal-only\ndescription: Internal skill\n---\nInternal content."
    )

    # Skill 4: experimental (NOT in marketplace)
    experimental_dir = skills_dir / "experimental"
    experimental_dir.mkdir()
    (experimental_dir / "SKILL.md").write_text(
        "---\nname: experimental\ndescription: Experimental\n---\nExperimental content."
    )

    # Create default marketplace with only git and docker
    marketplace = {
        "name": "default",
        "owner": {"name": "OpenHands", "email": "test@test.com"},
        "metadata": {"description": "Test marketplace", "version": "1.0.0"},
        "plugins": [
            {"name": "git", "source": "./git", "description": "Git skill"},
            {"name": "docker", "source": "./docker", "description": "Docker skill"},
        ],
    }
    (marketplaces_dir / "default.json").write_text(json.dumps(marketplace))

    # Create .git directory to simulate a git repo
    (repo_dir / ".git").mkdir()

    return repo_dir


def test_load_public_skills_filters_by_marketplace(mock_repo_with_marketplace):
    """Test that load_public_skills only loads skills listed in the marketplace."""
    marketplace_file = mock_repo_with_marketplace / "marketplaces" / "default.json"
    skills = load_public_skills(str(marketplace_file))

    # Should only have git and docker (from marketplace), not internal-only
    skill_names = {s.name for s in skills}
    assert skill_names == {"git", "docker"}
    assert "internal-only" not in skill_names
    assert "experimental" not in skill_names


def test_load_public_skills_returns_empty_when_no_marketplace(tmp_path):
    """Test that load_public_skills returns empty list when marketplace is missing."""
    # Create repo without marketplace
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create skills
    for name in ["git", "docker", "internal-only"]:
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\n{name} content."
        )

    (repo_dir / ".git").mkdir()

    def mock_update_repo(repo_url, branch, cache_dir):
        return repo_dir

    with (
        patch(
            "openhands.sdk.context.skills.skill.update_skills_repository",
            side_effect=mock_update_repo,
        ),
        patch(
            "openhands.sdk.context.skills.skill.get_skills_cache_dir",
            return_value=tmp_path,
        ),
    ):
        skills = load_public_skills()

        # Should have no skills since marketplace is missing
        assert skills == []


def test_load_public_skills_handles_skills_with_marketplace(tmp_path):
    """Test marketplace filtering works with different skill structures."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create skills in directory format with SKILL.md
    for name in ["git", "docker", "internal"]:
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ntriggers:\n  - {name}\n---\n{name} skill."
        )

    # Create marketplace that includes git and docker but not internal
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()
    marketplace = {
        "name": "default",
        "owner": {"name": "Test Team"},
        "plugins": [
            {"name": "git", "source": "./git"},
            {"name": "docker", "source": "./docker"},
        ],
    }
    marketplace_file = marketplaces_dir / "default.json"
    marketplace_file.write_text(json.dumps(marketplace))

    skills = load_public_skills(str(marketplace_file))

    # Should only have git and docker from marketplace
    skill_names = {s.name for s in skills}
    assert skill_names == {"git", "docker"}
    assert "internal" not in skill_names


def test_load_public_skills_with_custom_marketplace(tmp_path):
    """Test loading skills from different local marketplace files."""
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create skills
    for name in ["git", "docker", "internal-only", "experimental"]:
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\n{name} content."
        )

    # Create marketplaces directory with multiple marketplaces
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()

    # Default marketplace with git and docker
    default_marketplace = {
        "name": "default",
        "owner": {"name": "OpenHands", "email": "test@test.com"},
        "plugins": [
            {"name": "git", "source": "./git"},
            {"name": "docker", "source": "./docker"},
        ],
    }
    default_file = marketplaces_dir / "default.json"
    default_file.write_text(json.dumps(default_marketplace))

    # Custom marketplace with all skills
    custom_marketplace = {
        "name": "custom",
        "owner": {"name": "OpenHands", "email": "test@test.com"},
        "plugins": [
            {"name": "git", "source": "./git"},
            {"name": "docker", "source": "./docker"},
            {"name": "internal-only", "source": "./internal-only"},
            {"name": "experimental", "source": "./experimental"},
        ],
    }
    custom_file = marketplaces_dir / "custom.json"
    custom_file.write_text(json.dumps(custom_marketplace))

    # Default marketplace should only have git and docker
    skills_default = load_public_skills(str(default_file))
    skill_names_default = {s.name for s in skills_default}
    assert skill_names_default == {"git", "docker"}

    # Custom marketplace should have all skills
    skills_custom = load_public_skills(str(custom_file))
    skill_names_custom = {s.name for s in skills_custom}
    assert skill_names_custom == {"git", "docker", "internal-only", "experimental"}


def test_load_public_skills_from_local_path(tmp_path):
    """Test loading skills from a local marketplace file."""
    repo_dir = tmp_path / "local_skills"
    repo_dir.mkdir()
    skills_dir = repo_dir / "skills"
    skills_dir.mkdir()

    # Create skills
    for name in ["git", "docker"]:
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}\n---\n{name} content."
        )

    # Create marketplace
    marketplaces_dir = repo_dir / "marketplaces"
    marketplaces_dir.mkdir()
    marketplace = {
        "name": "default",
        "owner": {"name": "Test", "email": "test@test.com"},
        "plugins": [
            {"name": "git", "source": "./git"},
            {"name": "docker", "source": "./docker"},
        ],
    }
    marketplace_file = marketplaces_dir / "default.json"
    marketplace_file.write_text(json.dumps(marketplace))

    # Load from local marketplace file
    skills = load_public_skills(str(marketplace_file))
    skill_names = {s.name for s in skills}
    assert skill_names == {"git", "docker"}
