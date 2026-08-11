# Consumer development workflow

Shared numerical changes follow this sequence:

1. Add a characterization case in the consumer that needs the behavior.
2. Define the smallest schema-neutral array/model contract here.
3. Implement and test the pure numerical primitive.
4. Update the consumer adapter and runtime provenance.
5. Run the shared suite and the consumer compatibility suite.
6. Build a wheel and publish a versioned release.

Daily consumer environments install stable wheels. Editable installs are
limited to an isolated shared-development environment and must never be
required through `PYTHONPATH` or sibling source paths in production.

The shared package remains independent of file formats, plotting, simulator
APIs, and consumer workflow policy.
