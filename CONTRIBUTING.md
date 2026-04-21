# Contributing to PRISM

We welcome contributions to PRISM! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Abduep53/prism.git
   cd prism
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv prism_env
   source prism_env/bin/activate  # On Windows: prism_env\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the coding style guidelines below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### 4. Code Quality Checks
```bash
# Format code
black src/ examples/ tests/

# Lint code
flake8 src/ examples/ tests/

# Type checking
mypy src/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 🎯 Contribution Areas

### High Priority
- **Bug fixes** for existing functionality
- **Performance improvements** for training and inference
- **Documentation** improvements and examples
- **Test coverage** for existing modules

### Medium Priority
- **New privacy mechanisms** (local DP, secure aggregation)
- **Additional model architectures** (transformer-based, attention mechanisms)
- **Multi-modal support** (RGB + pose, audio + pose)
- **Deployment tools** (Docker, Kubernetes, edge deployment)

### Low Priority
- **Visualization tools** for privacy analysis
- **Benchmarking utilities** for different datasets
- **Educational materials** and tutorials
- **Integration examples** with popular frameworks

## 📋 Coding Standards

### Python Style
- Follow **PEP 8** style guidelines
- Use **type hints** for all function parameters and return values
- Write **docstrings** for all public functions and classes
- Keep functions **small and focused** (max 50 lines)
- Use **descriptive variable names**

### Code Organization
- Place new modules in the appropriate `src/` subdirectory
- Add corresponding tests in `tests/`
- Update `__init__.py` files to export public APIs
- Follow the existing project structure

### Documentation
- Update **README.md** for user-facing changes
- Add **docstrings** with examples for new functions
- Update **API documentation** for new modules
- Include **type information** in docstrings

### Testing Requirements
- **Unit tests** for all new functions and classes
- **Integration tests** for new features
- **Edge case testing** for error conditions
- **Performance tests** for critical paths
- Aim for **>90% test coverage**

## 🧪 Testing Guidelines

### Test Structure
```python
def test_function_name():
    """Test description explaining what is being tested."""
    # Arrange
    input_data = create_test_data()
    expected_output = expected_result()
    
    # Act
    actual_output = function_under_test(input_data)
    
    # Assert
    assert actual_output == expected_output
```

### Test Categories
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test module interactions
- **Privacy Tests**: Test differential privacy guarantees
- **Performance Tests**: Test latency and memory usage
- **Regression Tests**: Test against known issues

### Running Tests
```bash
# All tests
pytest

# Specific category
pytest -m "privacy"

# With verbose output
pytest -v

# Stop on first failure
pytest -x
```

## 🔒 Privacy Considerations

### Differential Privacy
- All privacy-preserving functions must maintain **ε-differential privacy**
- Include **privacy budget tracking** in new algorithms
- Test **privacy guarantees** with formal verification
- Document **privacy parameters** and their impact

### Data Handling
- Never commit **sensitive data** to the repository
- Use **synthetic data** for testing and examples
- Implement **data anonymization** for any real data
- Follow **data minimization** principles

### Security
- Validate all **input parameters** for security vulnerabilities
- Use **secure random number generation** for noise addition
- Implement **proper error handling** without information leakage
- Follow **secure coding practices**

## 📚 Documentation Standards

### Code Documentation
```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When invalid parameters are provided
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    pass
```

### README Updates
- Update **installation instructions** for new dependencies
- Add **usage examples** for new features
- Update **performance benchmarks** if applicable
- Include **privacy considerations** for new functionality

### API Documentation
- Document all **public functions and classes**
- Include **parameter types and descriptions**
- Provide **usage examples** for complex functions
- Update **changelog** for significant changes

## 🐛 Bug Reports

### Before Submitting
1. **Search existing issues** to avoid duplicates
2. **Test with latest version** to ensure bug still exists
3. **Gather information** about your environment
4. **Create minimal reproduction** if possible

### Bug Report Template
```markdown
**Bug Description**
Brief description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PRISM version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Before Submitting
1. **Check existing issues** for similar requests
2. **Consider the project scope** and goals
3. **Think about implementation** complexity
4. **Consider privacy implications**

### Feature Request Template
```markdown
**Feature Description**
Brief description of the requested feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
What other approaches were considered?

**Additional Context**
Any other relevant information
```

## 🔄 Pull Request Process

### Before Submitting
- [ ] **Tests pass** locally
- [ ] **Code is formatted** with black
- [ ] **Linting passes** with flake8
- [ ] **Type checking passes** with mypy
- [ ] **Documentation is updated**
- [ ] **Changelog is updated** (if applicable)

### PR Template
```markdown
**Description**
Brief description of changes

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally

**Privacy Impact**
- [ ] No privacy impact
- [ ] Privacy-preserving changes
- [ ] Privacy analysis required

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
```

### Review Process
1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Privacy review** for privacy-related changes
4. **Testing verification** for critical changes
5. **Documentation review** for user-facing changes

## 🏷️ Release Process

### Version Numbering
- **Major** (1.0.0): Breaking changes
- **Minor** (1.1.0): New features, backward compatible
- **Patch** (1.0.1): Bug fixes, backward compatible

### Release Checklist
- [ ] **All tests pass**
- [ ] **Documentation is complete**
- [ ] **Changelog is updated**
- [ ] **Version numbers are updated**
- [ ] **Release notes are written**
- [ ] **GitHub release is created**

## 🤝 Community Guidelines

### Code of Conduct
- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions

### Communication
- **GitHub Issues** for bug reports and feature requests
- **GitHub Discussions** for questions and general discussion
- **Pull Requests** for code contributions
- **Email** for security issues (abduyusufbek.2@gmail.com)

### Recognition
- Contributors will be **acknowledged** in the README
- Significant contributions will be **highlighted** in release notes
- Maintainers will be **recognized** in the project documentation

## 📞 Getting Help

### Documentation
- **README.md**: Quick start and overview
- **API Reference**: Detailed function documentation
- **Examples**: Usage examples and tutorials
- **Papers**: Scientific background and methodology

### Community Support
- **GitHub Discussions**: Ask questions and get help
- **GitHub Issues**: Report bugs and request features
- **Email**: Contact maintainers directly

### Development Support
- **Code Review**: Get feedback on your contributions
- **Mentoring**: Pair programming and guidance
- **Documentation**: Help improve project documentation

---

Thank you for contributing to PRISM! Your contributions help advance privacy-preserving AI for human behavior analysis. 🚀🔒🧠
